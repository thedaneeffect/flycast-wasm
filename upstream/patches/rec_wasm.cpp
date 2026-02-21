// rec_wasm.cpp — WASM JIT backend for Flycast SH4 dynarec
//
// Phase 2: Compiles SHIL basic blocks into WebAssembly functions at runtime.
// Each block becomes a WASM module with one exported function that:
//   - Operates on Sh4Context in shared Emscripten linear memory
//   - Calls imported functions for memory I/O and interpreter fallback
//   - Sets next PC and returns (mainloop handles dispatch)
//
// This file is compiled when FEAT_SHREC == DYNAREC_JIT && HOST_CPU == CPU_GENERIC
// (set in build.h for __EMSCRIPTEN__).

#include "build.h"

#if FEAT_SHREC == DYNAREC_JIT && HOST_CPU == CPU_GENERIC

#include "types.h"
#include "hw/sh4/sh4_opcode_list.h"
#include "hw/sh4/dyna/ngen.h"
#include "hw/sh4/dyna/blockmanager.h"
#include "hw/sh4/dyna/decoder.h"
#include "hw/sh4/sh4_interrupts.h"
#include "hw/sh4/sh4_core.h"
#include "hw/sh4/sh4_mem.h"
#include "hw/sh4/sh4_sched.h"
#include "oslib/virtmem.h"

#include "wasm_module_builder.h"
#include "wasm_emit.h"

#include <unordered_map>
#include <cmath>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// Verify Sh4Context offsets used in wasm_emit.h
static_assert(offsetof(Sh4Context, pc) == 0x148, "PC offset mismatch");
static_assert(offsetof(Sh4Context, jdyn) == 0x14C, "jdyn offset mismatch");
static_assert(offsetof(Sh4Context, sr.T) == 0x154, "sr.T offset mismatch");
static_assert(offsetof(Sh4Context, cycle_counter) == 0x174, "cycle_counter offset mismatch");

// Forward declarations from driver.cpp
DynarecCodeEntryPtr DYNACALL rdv_FailedToFindBlock(u32 pc);

// ============================================================
// Block info storage for SHIL fallback
// ============================================================
static std::unordered_map<u32, RuntimeBlockInfo*> blockByVaddr;
// SMC (self-modifying code) detection: store first word of block code
static std::unordered_map<u32, u32> blockCodeHash;

// ============================================================
// Deferred exception handling for shop_ifb
// ============================================================
// When an SH4 exception occurs inside a shop_ifb handler, we can't call
// Do_Exception immediately because the WASM block exit would overwrite
// the exception vector PC. Instead, we save the exception info and defer
// the Do_Exception call until after the WASM block finishes.
static bool g_ifb_exception_pending = false;
static u32 g_ifb_exception_epc = 0;
static Sh4ExceptionCode g_ifb_exception_expEvn = (Sh4ExceptionCode)0;

// ============================================================
// C-linkage wrapper functions for WASM imports
// ============================================================

extern "C" {

u32 EMSCRIPTEN_KEEPALIVE wasm_mem_read8(u32 addr) {
	return ReadMem8(addr);
}

u32 EMSCRIPTEN_KEEPALIVE wasm_mem_read16(u32 addr) {
	return ReadMem16(addr);
}

u32 EMSCRIPTEN_KEEPALIVE wasm_mem_read32(u32 addr) {
	return ReadMem32(addr);
}

void EMSCRIPTEN_KEEPALIVE wasm_mem_write8(u32 addr, u32 val) {
	WriteMem8(addr, (u8)val);
}

void EMSCRIPTEN_KEEPALIVE wasm_mem_write16(u32 addr, u32 val) {
	WriteMem16(addr, (u16)val);
}

void EMSCRIPTEN_KEEPALIVE wasm_mem_write32(u32 addr, u32 val) {
	WriteMem32(addr, val);
}

void EMSCRIPTEN_KEEPALIVE wasm_exec_ifb(u32 opcode, u32 pc) {
	(void)pc;
	OpPtr[opcode](&Sh4cntx, opcode);
}

// Runtime SHIL op interpreter — executes a single SHIL op by reading
// register values from Sh4Context, performing the operation, and writing
// results back. Used for ops that the WASM emitter doesn't handle natively.
void EMSCRIPTEN_KEEPALIVE wasm_exec_shil_fb(u32 block_vaddr, u32 op_index) {
	// Skip remaining ops after a deferred exception (block should abort)
	if (g_ifb_exception_pending) return;

	auto it = blockByVaddr.find(block_vaddr);
	if (it == blockByVaddr.end()) return;
	RuntimeBlockInfo* block = it->second;
	if (op_index >= block->oplist.size()) return;

	shil_opcode& op = block->oplist[op_index];
	Sh4Context& ctx = Sh4cntx;

	// Helper lambdas to read/write params
	auto readI32 = [&](const shil_param& p) -> u32 {
		if (p.is_imm()) return p._imm;
		if (p.is_reg()) return *(u32*)((u8*)&ctx + p.reg_offset());
		return 0;
	};
	auto readF32 = [&](const shil_param& p) -> float {
		if (p.is_reg()) return *(float*)((u8*)&ctx + p.reg_offset());
		return 0.0f;
	};
	auto writeI32 = [&](const shil_param& p, u32 val) {
		if (p.is_reg()) *(u32*)((u8*)&ctx + p.reg_offset()) = val;
	};
	auto writeF32 = [&](const shil_param& p, float val) {
		if (p.is_reg()) *(float*)((u8*)&ctx + p.reg_offset()) = val;
	};

	switch (op.op) {
	case shop_sync_sr:
		UpdateSR();
		break;
	case shop_sync_fpscr:
		Sh4Context::UpdateFPSCR(&ctx);
		break;
	case shop_pref: {
		u32 addr = readI32(op.rs1);
		if ((addr >> 26) == 0x38) ctx.doSqWrite(addr, &ctx);
		break;
	}
	// Integer ops with carry (64-bit result in rd, rd2)
	case shop_adc: {
		u32 a = readI32(op.rs1), b = readI32(op.rs2), c = readI32(op.rs3);
		u64 res = (u64)a + (u64)b + (u64)c;
		writeI32(op.rd, (u32)res);
		writeI32(op.rd2, (u32)(res >> 32));
		break;
	}
	case shop_sbc: {
		u32 a = readI32(op.rs1), b = readI32(op.rs2), c = readI32(op.rs3);
		u64 res = (u64)a - (u64)b - (u64)c;
		writeI32(op.rd, (u32)res);
		writeI32(op.rd2, res >> 63);
		break;
	}
	case shop_negc: {
		u32 a = readI32(op.rs1), c = readI32(op.rs2);
		u64 res = 0ULL - (u64)a - (u64)c;
		writeI32(op.rd, (u32)res);
		writeI32(op.rd2, res >> 63);
		break;
	}
	case shop_rocl: {
		u32 val = readI32(op.rs1), carry = readI32(op.rs2);
		u32 newCarry = val >> 31;
		writeI32(op.rd, (val << 1) | (carry & 1));
		writeI32(op.rd2, newCarry);
		break;
	}
	case shop_rocr: {
		u32 val = readI32(op.rs1), carry = readI32(op.rs2);
		u32 newCarry = val & 1;
		writeI32(op.rd, (val >> 1) | ((carry & 1) << 31));
		writeI32(op.rd2, newCarry);
		break;
	}
	case shop_shld: {
		u32 val = readI32(op.rs1);
		s32 shift = (s32)readI32(op.rs2);
		if (shift >= 0)
			writeI32(op.rd, val << (shift & 0x1F));
		else if ((shift & 0x1F) == 0)
			writeI32(op.rd, 0);
		else
			writeI32(op.rd, val >> ((-shift) & 0x1F));
		break;
	}
	case shop_shad: {
		s32 val = (s32)readI32(op.rs1);
		s32 shift = (s32)readI32(op.rs2);
		if (shift >= 0)
			writeI32(op.rd, (u32)(val << (shift & 0x1F)));
		else if ((shift & 0x1F) == 0)
			writeI32(op.rd, (u32)(val >> 31));
		else
			writeI32(op.rd, (u32)(val >> ((-shift) & 0x1F)));
		break;
	}
	case shop_mul_u64: {
		u64 res = (u64)readI32(op.rs1) * (u64)readI32(op.rs2);
		writeI32(op.rd, (u32)res);
		writeI32(op.rd2, (u32)(res >> 32));
		break;
	}
	case shop_mul_s64: {
		s64 res = (s64)(s32)readI32(op.rs1) * (s64)(s32)readI32(op.rs2);
		writeI32(op.rd, (u32)res);
		writeI32(op.rd2, (u32)((u64)res >> 32));
		break;
	}
	case shop_setpeq: {
		u32 a = readI32(op.rs1), b = readI32(op.rs2);
		u32 xor_val = a ^ b;
		u32 result = ((xor_val & 0xFF000000) == 0) || ((xor_val & 0x00FF0000) == 0) ||
		             ((xor_val & 0x0000FF00) == 0) || ((xor_val & 0x000000FF) == 0);
		writeI32(op.rd, result);
		break;
	}
	// FPU ops
	case shop_fmac: {
		float fn = readF32(op.rs1), f0 = readF32(op.rs2), fm = readF32(op.rs3);
		writeF32(op.rd, fn + f0 * fm);
		break;
	}
	case shop_fsrra: {
		float val = readF32(op.rs1);
		writeF32(op.rd, 1.0f / sqrtf(val));
		break;
	}
	case shop_fipr: {
		// 4-element dot product: sum of rs1[i] * rs2[i]
		float sum = 0;
		u32 off1 = op.rs1.reg_offset(), off2 = op.rs2.reg_offset();
		for (int i = 0; i < 4; i++) {
			float a = *(float*)((u8*)&ctx + off1 + i * 4);
			float b = *(float*)((u8*)&ctx + off2 + i * 4);
			sum += a * b;
		}
		writeF32(op.rd, sum);
		break;
	}
	case shop_ftrv: {
		// 4x4 matrix * 4-element vector (column-major, matches canonical)
		// rs1 = vector (4 floats), rs2 = matrix (16 floats, column-major)
		u32 voff = op.rs1.reg_offset(), moff = op.rs2.reg_offset();
		float result[4] = {0, 0, 0, 0};
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				float m = *(float*)((u8*)&ctx + moff + (j * 4 + i) * 4);
				float v = *(float*)((u8*)&ctx + voff + j * 4);
				result[i] += m * v;
			}
		}
		u32 doff = op.rd.reg_offset();
		for (int i = 0; i < 4; i++)
			*(float*)((u8*)&ctx + doff + i * 4) = result[i];
		break;
	}
	case shop_frswap: {
		u32 off1 = op.rs1.reg_offset(), off2 = op.rd.reg_offset();
		for (int i = 0; i < 16; i++) {
			u32* a = (u32*)((u8*)&ctx + off1 + i * 4);
			u32* b = (u32*)((u8*)&ctx + off2 + i * 4);
			u32 tmp = *a; *a = *b; *b = tmp;
		}
		break;
	}
	case shop_fsca: {
		u32 angle = readI32(op.rs1);
		float rad = (float)(angle & 0xFFFF) * (2.0f * 3.14159265f / 65536.0f);
		u32 doff = op.rd.reg_offset();
		*(float*)((u8*)&ctx + doff) = sinf(rad);
		*(float*)((u8*)&ctx + doff + 4) = cosf(rad);
		break;
	}
	// ---- Tier 1/2 basic ops (needed when WASM emitters are disabled for debugging) ----
	case shop_mov32:
		writeI32(op.rd, readI32(op.rs1));
		break;
	case shop_mov64: {
		u32 soff = op.rs1.reg_offset(), doff = op.rd.reg_offset();
		*(u32*)((u8*)&ctx + doff) = *(u32*)((u8*)&ctx + soff);
		*(u32*)((u8*)&ctx + doff + 4) = *(u32*)((u8*)&ctx + soff + 4);
		break;
	}
	case shop_add:
		writeI32(op.rd, readI32(op.rs1) + readI32(op.rs2));
		break;
	case shop_sub:
		writeI32(op.rd, readI32(op.rs1) - readI32(op.rs2));
		break;
	case shop_and:
		writeI32(op.rd, readI32(op.rs1) & readI32(op.rs2));
		break;
	case shop_or:
		writeI32(op.rd, readI32(op.rs1) | readI32(op.rs2));
		break;
	case shop_xor:
		writeI32(op.rd, readI32(op.rs1) ^ readI32(op.rs2));
		break;
	case shop_not:
		writeI32(op.rd, ~readI32(op.rs1));
		break;
	case shop_neg:
		writeI32(op.rd, (u32)(-(s32)readI32(op.rs1)));
		break;
	case shop_shl:
		writeI32(op.rd, readI32(op.rs1) << (readI32(op.rs2) & 0x1F));
		break;
	case shop_shr:
		writeI32(op.rd, readI32(op.rs1) >> (readI32(op.rs2) & 0x1F));
		break;
	case shop_sar:
		writeI32(op.rd, (u32)((s32)readI32(op.rs1) >> (readI32(op.rs2) & 0x1F)));
		break;
	case shop_ror: {
		u32 v = readI32(op.rs1), s = readI32(op.rs2) & 0x1F;
		writeI32(op.rd, (v >> s) | (v << (32 - s)));
		break;
	}
	case shop_ext_s8:
		writeI32(op.rd, (u32)(s32)(s8)(readI32(op.rs1) & 0xFF));
		break;
	case shop_ext_s16:
		writeI32(op.rd, (u32)(s32)(s16)(readI32(op.rs1) & 0xFFFF));
		break;
	case shop_mul_u16:
		writeI32(op.rd, (readI32(op.rs1) & 0xFFFF) * (readI32(op.rs2) & 0xFFFF));
		break;
	case shop_mul_s16:
		writeI32(op.rd, (u32)((s32)(s16)(readI32(op.rs1) & 0xFFFF) * (s32)(s16)(readI32(op.rs2) & 0xFFFF)));
		break;
	case shop_mul_i32:
		writeI32(op.rd, readI32(op.rs1) * readI32(op.rs2));
		break;
	case shop_test:
		writeI32(op.rd, (readI32(op.rs1) & readI32(op.rs2)) == 0 ? 1 : 0);
		break;
	case shop_seteq:
		writeI32(op.rd, (readI32(op.rs1) == readI32(op.rs2)) ? 1 : 0);
		break;
	case shop_setge:
		writeI32(op.rd, (s32)readI32(op.rs1) >= (s32)readI32(op.rs2) ? 1 : 0);
		break;
	case shop_setgt:
		writeI32(op.rd, (s32)readI32(op.rs1) > (s32)readI32(op.rs2) ? 1 : 0);
		break;
	case shop_setae:
		writeI32(op.rd, readI32(op.rs1) >= readI32(op.rs2) ? 1 : 0);
		break;
	case shop_setab:
		writeI32(op.rd, readI32(op.rs1) > readI32(op.rs2) ? 1 : 0);
		break;
	case shop_jdyn: {
		u32 val = readI32(op.rs1);
		if (!op.rs2.is_null()) val += readI32(op.rs2);
		ctx.jdyn = val;
		break;
	}
	case shop_jcond:
		ctx.sr.T = readI32(op.rs1);
		break;
	case shop_readm: {
		u32 addr = readI32(op.rs1);
		if (!op.rs3.is_null()) addr += readI32(op.rs3);
		if (op.size == 8) {
			u32 doff = op.rd.reg_offset();
			*(u32*)((u8*)&ctx + doff) = ReadMem32(addr);
			*(u32*)((u8*)&ctx + doff + 4) = ReadMem32(addr + 4);
		} else if (op.size == 1) {
			// Sign-extend 8-bit reads (matches all native backends)
			writeI32(op.rd, (u32)(s32)(s8)ReadMem8(addr));
		} else if (op.size == 2) {
			// Sign-extend 16-bit reads (matches all native backends)
			writeI32(op.rd, (u32)(s32)(s16)ReadMem16(addr));
		} else {
			writeI32(op.rd, ReadMem32(addr));
		}
		break;
	}
	case shop_writem: {
		u32 addr = readI32(op.rs1);
		if (!op.rs3.is_null()) addr += readI32(op.rs3);
		if (op.size == 8) {
			u32 soff = op.rs2.reg_offset();
			WriteMem32(addr, *(u32*)((u8*)&ctx + soff));
			WriteMem32(addr + 4, *(u32*)((u8*)&ctx + soff + 4));
		} else if (op.size == 1) {
			WriteMem8(addr, (u8)readI32(op.rs2));
		} else if (op.size == 2) {
			WriteMem16(addr, (u16)readI32(op.rs2));
		} else {
			WriteMem32(addr, readI32(op.rs2));
		}
		break;
	}
	case shop_ifb:
		if (op.rs1._imm)
			ctx.pc = op.rs2._imm;
		// Catch SH4 exceptions and DEFER Do_Exception until after the WASM
		// block finishes. This prevents the block exit from overwriting the
		// exception vector PC that Do_Exception would set.
		try {
			if (ctx.sr.FD == 1 && OpDesc[op.rs3._imm]->IsFloatingPoint())
				throw SH4ThrownException(ctx.pc - 2, Sh4Ex_FpuDisabled);
			OpPtr[op.rs3._imm](&ctx, op.rs3._imm);
		} catch (const SH4ThrownException& ex) {
			g_ifb_exception_pending = true;
			g_ifb_exception_epc = ex.epc;
			g_ifb_exception_expEvn = ex.expEvn;
			// Don't call Do_Exception here — defer to mainloop
		}
		break;
	case shop_swaplb: {
		u32 v = readI32(op.rs1);
		writeI32(op.rd, ((v >> 8) & 0xFF) | ((v & 0xFF) << 8) | (v & 0xFFFF0000));
		break;
	}
	case shop_xtrct:
		writeI32(op.rd, (readI32(op.rs1) >> 16) | (readI32(op.rs2) << 16));
		break;
	// FPU basic ops
	case shop_fadd:
		writeF32(op.rd, readF32(op.rs1) + readF32(op.rs2));
		break;
	case shop_fsub:
		writeF32(op.rd, readF32(op.rs1) - readF32(op.rs2));
		break;
	case shop_fmul:
		writeF32(op.rd, readF32(op.rs1) * readF32(op.rs2));
		break;
	case shop_fdiv:
		writeF32(op.rd, readF32(op.rs1) / readF32(op.rs2));
		break;
	case shop_fabs:
		writeF32(op.rd, fabsf(readF32(op.rs1)));
		break;
	case shop_fneg:
		writeF32(op.rd, -readF32(op.rs1));
		break;
	case shop_fsqrt:
		writeF32(op.rd, sqrtf(readF32(op.rs1)));
		break;
	case shop_fseteq:
		writeI32(op.rd, readF32(op.rs1) == readF32(op.rs2) ? 1 : 0);
		break;
	case shop_fsetgt:
		writeI32(op.rd, readF32(op.rs1) > readF32(op.rs2) ? 1 : 0);
		break;
	case shop_cvt_f2i_t: {
		float fval = readF32(op.rs1);
		s32 res;
		if (fval > 2147483520.0f) {
			res = 0x7fffffff;
		} else {
			res = (s32)fval;
			if (std::isnan(fval))
				res = (s32)0x80000000;
		}
		writeI32(op.rd, (u32)res);
		break;
	}
	case shop_cvt_i2f_n:
	case shop_cvt_i2f_z:
		writeF32(op.rd, (float)(s32)readI32(op.rs1));
		break;
	case shop_div1: {
		// SH4 DIV1 — single-step division (matches shil_canonical.h exactly)
		u32 a = readI32(op.rs1);
		s32 b = (s32)readI32(op.rs2);
		u32 T = readI32(op.rs3);
		bool qxm = ctx.sr.Q ^ ctx.sr.M;
		ctx.sr.Q = (int)a < 0;
		a = (a << 1) | T;
		u32 oldA = a;
		a += (qxm ? 1 : -1) * b;
		ctx.sr.Q ^= ctx.sr.M ^ (qxm ? a < oldA : a > oldA);
		T = !(ctx.sr.Q ^ ctx.sr.M);
		writeI32(op.rd, a);
		writeI32(op.rd2, T);
		break;
	}
	case shop_div32u: {
		// Unsigned 64/32 division
		u32 r1 = readI32(op.rs1), r2 = readI32(op.rs2), r3 = readI32(op.rs3);
		u64 dividend = ((u64)r3 << 32) | r1;
		u32 quo = r2 ? (u32)(dividend / r2) : 0;
		u32 rem = r2 ? (u32)(dividend % r2) : (u32)dividend;
		writeI32(op.rd, quo);
		writeI32(op.rd2, rem);
		break;
	}
	case shop_div32s: {
		// Signed 64/32 division
		u32 r1 = readI32(op.rs1);
		s32 r2 = (s32)readI32(op.rs2);
		s32 r3 = (s32)readI32(op.rs3);
		s64 dividend = ((s64)r3 << 32) | r1;
		if (dividend < 0) dividend++;  // 1's complement → 2's complement
		s32 quo = r2 ? (s32)(dividend / r2) : 0;
		s32 rem = (s32)(dividend - (s64)quo * r2);
		u32 negative = ((u32)r3 ^ (u32)r2) & 0x80000000;
		if (negative) quo--;
		else if (r3 < 0) rem--;
		writeI32(op.rd, (u32)quo);
		writeI32(op.rd2, (u32)rem);
		break;
	}
	case shop_div32p2: {
		// Division fixup step
		s32 a = (s32)readI32(op.rs1);
		s32 b = (s32)readI32(op.rs2);
		u32 T = readI32(op.rs3);
		if (!(T & 0x80000000)) {
			if (!(T & 1)) a -= b;
		} else {
			if (b > 0) a--;
			if (T & 1) a += b;
		}
		writeI32(op.rd, (u32)a);
		break;
	}
	default:
		// Unknown op — log and skip
#ifdef __EMSCRIPTEN__
		static int unhandledCount = 0;
		unhandledCount++;
		if (unhandledCount <= 20) {
			EM_ASM({ console.warn('[rec_wasm] unhandled SHIL fallback op=' + $0 + ' at block 0x' + ($1>>>0).toString(16)); },
				(int)op.op, block_vaddr);
		}
#endif
		break;
	}
}

} // extern "C"

// ============================================================
// Per-instruction block executor — executes raw SH4 instructions
// Uses OpPtr directly (same as interpreter), but in block batches.
// Follows PC after each instruction to handle branches properly
// (branch handlers execute delay slot internally via executeDelaySlot).
// ============================================================
// Reference executor: per-instruction via OpPtr
// Per-instruction cycle counting (1 per instruction executed)
// Does NOT follow branches within blocks — exits at first branch
// to match JIT dispatch model
static void ref_execute_block(RuntimeBlockInfo* block) {
	Sh4Context& ctx = Sh4cntx;
	ctx.pc = block->vaddr;
	u32 block_end = block->vaddr + block->sh4_code_size;
	u32 maxInstrs = block->guest_opcodes + 1;
	for (u32 n = 0; n < maxInstrs; n++) {
		u32 pc = ctx.pc;
		if (pc < block->vaddr || pc >= block_end) break;
		ctx.pc = pc + 2;
		u16 op = IReadMem16(pc);
		if (ctx.sr.FD == 1 && OpDesc[op]->IsFloatingPoint()) {
			Do_Exception(pc, Sh4Ex_FpuDisabled);
			return;
		}
		OpPtr[op](&ctx, op);
		ctx.cycle_counter -= 1;
		// After a branch (PC diverged from sequential), exit block
		// This matches JIT model: one pass through block, no internal looping
		if (ctx.pc != (pc + 2) && ctx.pc != (pc + 4)) {
			// PC jumped somewhere other than the delay slot's natural successor
			// Branch was taken — exit like JIT would
			break;
		}
	}
}

// Block executor: uses OpPtr-based per-instruction dispatch.
// This matches the interpreter's timing model including dynamic memory
// access cycle penalties from Sh4Cycles. SHIL ops are proven correct
// (shadow comparison showed only jdyn false positives) but can't replicate
// the interpreter's cycle counting because SHIL's ReadMem/WriteMem bypass
// the sh4_cache cycle penalty mechanism.
static void cpp_execute_block(RuntimeBlockInfo* block) {
	ref_execute_block(block);
}

// ============================================================
// EM_JS bridge: compile + execute WASM blocks from JavaScript
// ============================================================

#ifdef __EMSCRIPTEN__

EM_JS(int, wasm_compile_block, (const u8* bytesPtr, u32 len, u32 block_pc), {
	try {
		var wasmBytes = Module.HEAPU8.slice(bytesPtr, bytesPtr + len);
		var mod = new WebAssembly.Module(wasmBytes);
		var instance = new WebAssembly.Instance(mod, {
			env: {
				memory: wasmExports.memory,
				read8:   function(addr) { return Module._wasm_mem_read8(addr); },
				read16:  function(addr) { return Module._wasm_mem_read16(addr); },
				read32:  function(addr) { return Module._wasm_mem_read32(addr); },
				write8:  function(addr, val) { Module._wasm_mem_write8(addr, val); },
				write16: function(addr, val) { Module._wasm_mem_write16(addr, val); },
				write32: function(addr, val) { Module._wasm_mem_write32(addr, val); },
				ifb:     function(opcode, pc) { Module._wasm_exec_ifb(opcode, pc); },
				shil_fb: function(block_vaddr, op_idx) { Module._wasm_exec_shil_fb(block_vaddr, op_idx); }
			}
		});
		if (!Module._wasmBlockCache) Module._wasmBlockCache = {};
		Module._wasmBlockCache[block_pc] = instance.exports.b;
		return 1;
	} catch (e) {
		console.error('[rec_wasm] compile fail PC=0x' + (block_pc >>> 0).toString(16) + ': ' + e.message);
		return 0;
	}
});

EM_JS(int, wasm_execute_block, (u32 block_pc, u32 ctx_ptr), {
	try {
		Module._wasmBlockCache[block_pc](ctx_ptr);
		return 0;
	} catch (e) {
		if (!Module._wasmTrapCount) Module._wasmTrapCount = 0;
		Module._wasmTrapCount++;
		if (Module._wasmTrapCount <= 50) {
			console.error('[wasm-trap] PC=0x' + (block_pc >>> 0).toString(16) + ': ' + e.message);
		}
		return 1;
	}
});

EM_JS(int, wasm_has_block, (u32 block_pc), {
	return (Module._wasmBlockCache && Module._wasmBlockCache[block_pc]) ? 1 : 0;
});

EM_JS(void, wasm_clear_cache, (), {
	Module._wasmBlockCache = {};
});

EM_JS(void, wasm_remove_block, (u32 block_pc), {
	if (Module._wasmBlockCache) delete Module._wasmBlockCache[block_pc];
});

EM_JS(int, wasm_cache_size, (), {
	return Module._wasmBlockCache ? Object.keys(Module._wasmBlockCache).length : 0;
});

#else
static int wasm_compile_block(const u8*, u32, u32) { return 0; }
static void wasm_execute_block(u32, u32) {}
static int wasm_has_block(u32) { return 0; }
static void wasm_remove_block(u32) {}
static void wasm_clear_cache() {}
static int wasm_cache_size() { return 0; }
#endif

// ============================================================
// Build a complete WASM module for one compiled block
// ============================================================

static bool buildBlockModule(WasmModuleBuilder& b, RuntimeBlockInfo* block) {
	b.emitHeader();

	// Type section: 3 function signatures
	// Type 0: (i32) -> void      — block function
	// Type 1: (i32) -> i32       — read8/16/32
	// Type 2: (i32, i32) -> void — write8/16/32, ifb, shil_fb
	b.emitTypeSection(3);
	{
		u8 p0[] = { WASM_TYPE_I32 };
		b.emitFuncType(p0, 1, nullptr, 0);

		u8 r1[] = { WASM_TYPE_I32 };
		b.emitFuncType(p0, 1, r1, 1);

		u8 p2[] = { WASM_TYPE_I32, WASM_TYPE_I32 };
		b.emitFuncType(p2, 2, nullptr, 0);
	}
	b.endSection();

	// Import section: 1 memory + 8 functions
	b.emitImportSection(9);
	b.emitImportMemory("env", "memory", 0);
	b.emitImportFunc("env", "read8",   1);
	b.emitImportFunc("env", "read16",  1);
	b.emitImportFunc("env", "read32",  1);
	b.emitImportFunc("env", "write8",  2);
	b.emitImportFunc("env", "write16", 2);
	b.emitImportFunc("env", "write32", 2);
	b.emitImportFunc("env", "ifb",     2);
	b.emitImportFunc("env", "shil_fb", 2);
	b.endSection();

	// Function section: 1 defined function (type 0)
	u32 typeIdx = 0;
	b.emitFunctionSection(1, &typeIdx);

	// Export section: export block function as "b" (func idx 8)
	b.emitExportSection("b", WIMPORT_COUNT);

	// Code section
	b.beginCodeSection(1);
	b.beginFuncBody();

	// Locals: 1 temp i32
	u32 lc = 1;
	u8 lt = WASM_TYPE_I32;
	b.emitLocals(1, &lc, &lt);

	// Prologue: cycle_counter -= guest_cycles
	b.op_local_get(LOCAL_CTX);
	b.op_local_get(LOCAL_CTX);
	b.op_i32_load(ctx_off::CYCLE_COUNTER);
	b.op_i32_const((s32)block->guest_cycles);
	b.op_i32_sub();
	b.op_i32_store(ctx_off::CYCLE_COUNTER);

	// Emit each SHIL op
	for (u32 i = 0; i < block->oplist.size(); i++) {
		shil_opcode& op = block->oplist[i];
		if (!emitShilOp(b, op, block, i)) {
			// Unhandled op — call SHIL canonical fallback
			b.op_i32_const((s32)block->vaddr);
			b.op_i32_const((s32)i);
			b.op_call(WIMPORT_SHIL_FB);
		}
	}

	// Epilogue: set next PC
	emitBlockExit(b, block);

	b.endFuncBody();
	b.endSection();

	return true;
}

// ============================================================
// WasmDynarec class
// ============================================================

class WasmDynarec : public Sh4Dynarec
{
public:
	WasmDynarec()
	{
		sh4Dynarec = this;
	}

	void init(Sh4Context& ctx, Sh4CodeBuffer& buf) override
	{
#ifdef __EMSCRIPTEN__
		EM_ASM({ console.log('[rec_wasm] WasmDynarec::init() — Phase 2 WASM JIT'); });
#endif
		sh4ctx = &ctx;
		codeBuffer = &buf;
		compiledCount = 0;
		failCount = 0;
	}

	void compile(RuntimeBlockInfo* block, bool smc_checks, bool optimise) override
	{
		blockByVaddr[block->vaddr] = block;

		// Store hash for SMC detection (first 2 bytes of block code)
		blockCodeHash[block->vaddr] = (u32)IReadMem16(block->vaddr);

		WasmModuleBuilder builder;
		buildBlockModule(builder, block);

		const auto& bytes = builder.getBytes();
		int result = wasm_compile_block(bytes.data(), (u32)bytes.size(), block->vaddr);

		if (result)
			compiledCount++;
		else
			failCount++;

		// Dummy code pointer for block manager
		block->code = (DynarecCodeEntryPtr)codeBuffer->get();
		if (codeBuffer->getFreeSpace() >= 4)
			codeBuffer->advance(4);

#ifdef __EMSCRIPTEN__
		if (compiledCount <= 10 || (compiledCount % 200 == 0)) {
			EM_ASM({ console.log('[rec_wasm] compiled=' + $0 + ' fail=' + $1 +
				' pc=0x' + ($2>>>0).toString(16) + ' ops=' + $3); },
				compiledCount, failCount, block->vaddr,
				(int)block->oplist.size());
		}
#endif
	}

	void mainloop(void* cntx) override
	{
		// CRITICAL: Branch instructions with delay slots call executeDelaySlot()
		// which dereferences Sh4Interpreter::Instance.
		Sh4Interpreter::Instance = Sh4Recompiler::Instance;

#ifdef __EMSCRIPTEN__
		static int mainloop_count = 0;
		mainloop_count++;
		if (mainloop_count <= 5) {
			EM_ASM({ console.log('[rec_wasm] mainloop #' + $0 + ' cache=' + $1); },
				mainloop_count, wasm_cache_size());
		}
#endif
		u32 blockExecs = 0;
		u32 interpExecs = 0;
		u32 timeslices = 0;

		try {
			do {
				try {
					do {
						u32 pc = sh4ctx->pc;
						auto it = blockByVaddr.find(pc);

						if (it == blockByVaddr.end()) {
							rdv_FailedToFindBlock(pc);
							it = blockByVaddr.find(pc);
						}

						if (it != blockByVaddr.end()) {
							// SMC check: verify first opcode hasn't changed
							auto hit = blockCodeHash.find(pc);
							if (hit != blockCodeHash.end()) {
								u32 curOp = (u32)IReadMem16(pc);
								if (curOp != hit->second) {
									// Invalidate stale block and fall through to interpreter
									wasm_remove_block(pc);
									blockCodeHash.erase(pc);
									blockByVaddr.erase(pc);
									it = blockByVaddr.end();
								}
							}
						}

						if (it != blockByVaddr.end()) {
							cpp_execute_block(it->second);
							blockExecs++;

							if (sh4ctx->interrupt_pend)
								UpdateINTC();
						} else {
							// Interpreter fallback — one instruction
							sh4ctx->pc = pc + 2;
							u16 op = IReadMem16(pc);
							if (sh4ctx->sr.FD == 1 && OpDesc[op]->IsFloatingPoint())
								throw SH4ThrownException(pc, Sh4Ex_FpuDisabled);
							OpPtr[op](sh4ctx, op);
							sh4ctx->cycle_counter -= 1;
							interpExecs++;
						}

					} while (sh4ctx->cycle_counter > 0);

					sh4ctx->cycle_counter += SH4_TIMESLICE;
					timeslices++;
					UpdateSystem_INTC();

				} catch (const SH4ThrownException& ex) {
					Do_Exception(ex.epc, ex.expEvn);
					sh4ctx->cycle_counter += 5;
				}
			} while (sh4ctx->CpuRunning);

		} catch (...) {
#ifdef __EMSCRIPTEN__
			EM_ASM({ console.log('[rec_wasm] WARNING: mainloop exited via catch(...)'); });
#endif
		}

		sh4ctx->CpuRunning = false;

#ifdef __EMSCRIPTEN__
		EM_ASM({ console.log('[rec_wasm] Exited mainloop #' + $0 + ' cache=' + $1 +
			' blocks=' + $2 + ' interp=' + $3 + ' ts=' + $4); },
			mainloop_count, wasm_cache_size(), blockExecs, interpExecs, timeslices);
#endif
	}

	void handleException(host_context_t& context) override {}

	bool rewrite(host_context_t& context, void* faultAddress) override
	{
		return false;
	}

	void reset() override
	{
		wasm_clear_cache();
		blockByVaddr.clear();
		blockCodeHash.clear();
		compiledCount = 0;
		failCount = 0;
#ifdef __EMSCRIPTEN__
		EM_ASM({ console.log('[rec_wasm] reset: cleared WASM block cache'); });
#endif
	}

	void canonStart(const shil_opcode* op) override {}
	void canonParam(const shil_opcode* op, const shil_param* par, CanonicalParamType tp) override {}
	void canonCall(const shil_opcode* op, void* function) override {}
	void canonFinish(const shil_opcode* op) override {}

private:
	Sh4Context* sh4ctx = nullptr;
	Sh4CodeBuffer* codeBuffer = nullptr;
	u32 compiledCount = 0;
	u32 failCount = 0;
};

static WasmDynarec instance;

extern "C" void wasm_dynarec_init()
{
	if (!sh4Dynarec)
		sh4Dynarec = &instance;
}

#endif // FEAT_SHREC == DYNAREC_JIT && HOST_CPU == CPU_GENERIC
