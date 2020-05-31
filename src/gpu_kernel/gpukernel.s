	.text
	.file	"gpukernel.cpp"
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function __cxx_global_var_init
	.type	__cxx_global_var_init,@function
__cxx_global_var_init:                  # @__cxx_global_var_init
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movabsq	$_ZStL8__ioinit, %rdi
	callq	_ZNSt8ios_base4InitC1Ev
	movabsq	$_ZNSt8ios_base4InitD1Ev, %rax
	movq	%rax, %rdi
	movabsq	$_ZStL8__ioinit, %rsi
	movabsq	$__dso_handle, %rdx
	callq	__cxa_atexit
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	__cxx_global_var_init, .Lfunc_end0-__cxx_global_var_init
	.cfi_endproc
                                        # -- End function
	.text
	.weak	_ZN6Kalmar5indexILi1EEC2Ev # -- Begin function _ZN6Kalmar5indexILi1EEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC2Ev,@function
_ZN6Kalmar5indexILi1EEC2Ev:             # @_ZN6Kalmar5indexILi1EEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_ZN6Kalmar5indexILi1EEC2Ev, .Lfunc_end1-_ZN6Kalmar5indexILi1EEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	xorl	%esi, %esi
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev, .Lfunc_end2-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC1Ev # -- Begin function _ZN6Kalmar5indexILi1EEC1Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC1Ev,@function
_ZN6Kalmar5indexILi1EEC1Ev:             # @_ZN6Kalmar5indexILi1EEC1Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar5indexILi1EEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	_ZN6Kalmar5indexILi1EEC1Ev, .Lfunc_end3-_ZN6Kalmar5indexILi1EEC1Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC2ERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEC2ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC2ERKS1_,@function
_ZN6Kalmar5indexILi1EEC2ERKS1_:         # @_ZN6Kalmar5indexILi1EEC2ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end4:
	.size	_ZN6Kalmar5indexILi1EEC2ERKS1_, .Lfunc_end4-_ZN6Kalmar5indexILi1EEC2ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rax
	movq	%rdi, -24(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-24(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end5:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_, .Lfunc_end5-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2ERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC1ERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEC1ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC1ERKS1_,@function
_ZN6Kalmar5indexILi1EEC1ERKS1_:         # @_ZN6Kalmar5indexILi1EEC1ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi1EEC2ERKS1_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end6:
	.size	_ZN6Kalmar5indexILi1EEC1ERKS1_, .Lfunc_end6-_ZN6Kalmar5indexILi1EEC1ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC2Ei # -- Begin function _ZN6Kalmar5indexILi1EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC2Ei,@function
_ZN6Kalmar5indexILi1EEC2Ei:             # @_ZN6Kalmar5indexILi1EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end7:
	.size	_ZN6Kalmar5indexILi1EEC2Ei, .Lfunc_end7-_ZN6Kalmar5indexILi1EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end8:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei, .Lfunc_end8-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC1Ei # -- Begin function _ZN6Kalmar5indexILi1EEC1Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC1Ei,@function
_ZN6Kalmar5indexILi1EEC1Ei:             # @_ZN6Kalmar5indexILi1EEC1Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar5indexILi1EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end9:
	.size	_ZN6Kalmar5indexILi1EEC1Ei, .Lfunc_end9-_ZN6Kalmar5indexILi1EEC1Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC2EPKi # -- Begin function _ZN6Kalmar5indexILi1EEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC2EPKi,@function
_ZN6Kalmar5indexILi1EEC2EPKi:           # @_ZN6Kalmar5indexILi1EEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end10:
	.size	_ZN6Kalmar5indexILi1EEC2EPKi, .Lfunc_end10-_ZN6Kalmar5indexILi1EEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movl	(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end11:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi, .Lfunc_end11-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC1EPKi # -- Begin function _ZN6Kalmar5indexILi1EEC1EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC1EPKi,@function
_ZN6Kalmar5indexILi1EEC1EPKi:           # @_ZN6Kalmar5indexILi1EEC1EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi1EEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end12:
	.size	_ZN6Kalmar5indexILi1EEC1EPKi, .Lfunc_end12-_ZN6Kalmar5indexILi1EEC1EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC2EPi # -- Begin function _ZN6Kalmar5indexILi1EEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC2EPi,@function
_ZN6Kalmar5indexILi1EEC2EPi:            # @_ZN6Kalmar5indexILi1EEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end13:
	.size	_ZN6Kalmar5indexILi1EEC2EPi, .Lfunc_end13-_ZN6Kalmar5indexILi1EEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rcx
	movl	(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end14:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi, .Lfunc_end14-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEC1EPi # -- Begin function _ZN6Kalmar5indexILi1EEC1EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEC1EPi,@function
_ZN6Kalmar5indexILi1EEC1EPi:            # @_ZN6Kalmar5indexILi1EEC1EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi1EEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end15:
	.size	_ZN6Kalmar5indexILi1EEC1EPi, .Lfunc_end15-_ZN6Kalmar5indexILi1EEC1EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEaSERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEaSERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEaSERKS1_,@function
_ZN6Kalmar5indexILi1EEaSERKS1_:         # @_ZN6Kalmar5indexILi1EEaSERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end16:
	.size	_ZN6Kalmar5indexILi1EEaSERKS1_, .Lfunc_end16-_ZN6Kalmar5indexILi1EEaSERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEaSEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end17:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_, .Lfunc_end17-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi1EEixEj # -- Begin function _ZNK6Kalmar5indexILi1EEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi1EEixEj,@function
_ZNK6Kalmar5indexILi1EEixEj:            # @_ZNK6Kalmar5indexILi1EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end18:
	.size	_ZNK6Kalmar5indexILi1EEixEj, .Lfunc_end18-_ZNK6Kalmar5indexILi1EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj # -- Begin function _ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj,@function
_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj: # @_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end19:
	.size	_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj, .Lfunc_end19-_ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEixEj # -- Begin function _ZN6Kalmar5indexILi1EEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEixEj,@function
_ZN6Kalmar5indexILi1EEixEj:             # @_ZN6Kalmar5indexILi1EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end20:
	.size	_ZN6Kalmar5indexILi1EEixEj, .Lfunc_end20-_ZN6Kalmar5indexILi1EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EE3getEv
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end21:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj, .Lfunc_end21-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi1EEeqERKS1_ # -- Begin function _ZNK6Kalmar5indexILi1EEeqERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi1EEeqERKS1_,@function
_ZNK6Kalmar5indexILi1EEeqERKS1_:        # @_ZNK6Kalmar5indexILi1EEeqERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end22:
	.size	_ZNK6Kalmar5indexILi1EEeqERKS1_, .Lfunc_end22-_ZNK6Kalmar5indexILi1EEeqERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	xorl	%eax, %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	%eax, %esi
	callq	_ZNK6Kalmar5indexILi1EEixEj
	xorl	%esi, %esi
	movq	-16(%rbp), %rdi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi1EEixEj
	movl	-20(%rbp), %ecx         # 4-byte Reload
	cmpl	%eax, %ecx
	sete	%dl
	andb	$1, %dl
	movzbl	%dl, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end23:
	.size	_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_, .Lfunc_end23-_ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi1EEneERKS1_ # -- Begin function _ZNK6Kalmar5indexILi1EEneERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi1EEneERKS1_,@function
_ZNK6Kalmar5indexILi1EEneERKS1_:        # @_ZNK6Kalmar5indexILi1EEneERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZNK6Kalmar5indexILi1EEeqERKS1_
	xorb	$-1, %al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end24:
	.size	_ZNK6Kalmar5indexILi1EEneERKS1_, .Lfunc_end24-_ZNK6Kalmar5indexILi1EEneERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEpLERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEpLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEpLERKS1_,@function
_ZN6Kalmar5indexILi1EEpLERKS1_:         # @_ZN6Kalmar5indexILi1EEpLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end25:
	.size	_ZN6Kalmar5indexILi1EEpLERKS1_, .Lfunc_end25-_ZN6Kalmar5indexILi1EEpLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end26:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_, .Lfunc_end26-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmIERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEmIERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmIERKS1_,@function
_ZN6Kalmar5indexILi1EEmIERKS1_:         # @_ZN6Kalmar5indexILi1EEmIERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end27:
	.size	_ZN6Kalmar5indexILi1EEmIERKS1_, .Lfunc_end27-_ZN6Kalmar5indexILi1EEmIERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end28:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_, .Lfunc_end28-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmLERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEmLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmLERKS1_,@function
_ZN6Kalmar5indexILi1EEmLERKS1_:         # @_ZN6Kalmar5indexILi1EEmLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end29:
	.size	_ZN6Kalmar5indexILi1EEmLERKS1_, .Lfunc_end29-_ZN6Kalmar5indexILi1EEmLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end30:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_, .Lfunc_end30-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEdVERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EEdVERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEdVERKS1_,@function
_ZN6Kalmar5indexILi1EEdVERKS1_:         # @_ZN6Kalmar5indexILi1EEdVERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end31:
	.size	_ZN6Kalmar5indexILi1EEdVERKS1_, .Lfunc_end31-_ZN6Kalmar5indexILi1EEdVERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end32:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_, .Lfunc_end32-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EErMERKS1_ # -- Begin function _ZN6Kalmar5indexILi1EErMERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EErMERKS1_,@function
_ZN6Kalmar5indexILi1EErMERKS1_:         # @_ZN6Kalmar5indexILi1EErMERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end33:
	.size	_ZN6Kalmar5indexILi1EErMERKS1_, .Lfunc_end33-_ZN6Kalmar5indexILi1EErMERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end34:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_, .Lfunc_end34-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEpLEi # -- Begin function _ZN6Kalmar5indexILi1EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEpLEi,@function
_ZN6Kalmar5indexILi1EEpLEi:             # @_ZN6Kalmar5indexILi1EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end35:
	.size	_ZN6Kalmar5indexILi1EEpLEi, .Lfunc_end35-_ZN6Kalmar5indexILi1EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end36:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi, .Lfunc_end36-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmIEi # -- Begin function _ZN6Kalmar5indexILi1EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmIEi,@function
_ZN6Kalmar5indexILi1EEmIEi:             # @_ZN6Kalmar5indexILi1EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end37:
	.size	_ZN6Kalmar5indexILi1EEmIEi, .Lfunc_end37-_ZN6Kalmar5indexILi1EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end38:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi, .Lfunc_end38-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmLEi # -- Begin function _ZN6Kalmar5indexILi1EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmLEi,@function
_ZN6Kalmar5indexILi1EEmLEi:             # @_ZN6Kalmar5indexILi1EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end39:
	.size	_ZN6Kalmar5indexILi1EEmLEi, .Lfunc_end39-_ZN6Kalmar5indexILi1EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end40:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi, .Lfunc_end40-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEdVEi # -- Begin function _ZN6Kalmar5indexILi1EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEdVEi,@function
_ZN6Kalmar5indexILi1EEdVEi:             # @_ZN6Kalmar5indexILi1EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end41:
	.size	_ZN6Kalmar5indexILi1EEdVEi, .Lfunc_end41-_ZN6Kalmar5indexILi1EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end42:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi, .Lfunc_end42-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EErMEi # -- Begin function _ZN6Kalmar5indexILi1EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EErMEi,@function
_ZN6Kalmar5indexILi1EErMEi:             # @_ZN6Kalmar5indexILi1EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end43:
	.size	_ZN6Kalmar5indexILi1EErMEi, .Lfunc_end43-_ZN6Kalmar5indexILi1EErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end44:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi, .Lfunc_end44-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEppEv # -- Begin function _ZN6Kalmar5indexILi1EEppEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEppEv,@function
_ZN6Kalmar5indexILi1EEppEv:             # @_ZN6Kalmar5indexILi1EEppEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end45:
	.size	_ZN6Kalmar5indexILi1EEppEv, .Lfunc_end45-_ZN6Kalmar5indexILi1EEppEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEppEi # -- Begin function _ZN6Kalmar5indexILi1EEppEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEppEi,@function
_ZN6Kalmar5indexILi1EEppEi:             # @_ZN6Kalmar5indexILi1EEppEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi1EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end46:
	.size	_ZN6Kalmar5indexILi1EEppEi, .Lfunc_end46-_ZN6Kalmar5indexILi1EEppEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmmEv # -- Begin function _ZN6Kalmar5indexILi1EEmmEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmmEv,@function
_ZN6Kalmar5indexILi1EEmmEv:             # @_ZN6Kalmar5indexILi1EEmmEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end47:
	.size	_ZN6Kalmar5indexILi1EEmmEv, .Lfunc_end47-_ZN6Kalmar5indexILi1EEmmEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi1EEmmEi # -- Begin function _ZN6Kalmar5indexILi1EEmmEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi1EEmmEi,@function
_ZN6Kalmar5indexILi1EEmmEi:             # @_ZN6Kalmar5indexILi1EEmmEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi1EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end48:
	.size	_ZN6Kalmar5indexILi1EEmmEi, .Lfunc_end48-_ZN6Kalmar5indexILi1EEmmEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC2Ev # -- Begin function _ZN6Kalmar5indexILi2EEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC2Ev,@function
_ZN6Kalmar5indexILi2EEC2Ev:             # @_ZN6Kalmar5indexILi2EEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end49:
	.size	_ZN6Kalmar5indexILi2EEC2Ev, .Lfunc_end49-_ZN6Kalmar5indexILi2EEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	xorl	%esi, %esi
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	%rcx, %rdi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	xorl	%esi, %esi
	movq	-16(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end50:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev, .Lfunc_end50-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC1Ev # -- Begin function _ZN6Kalmar5indexILi2EEC1Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC1Ev,@function
_ZN6Kalmar5indexILi2EEC1Ev:             # @_ZN6Kalmar5indexILi2EEC1Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar5indexILi2EEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end51:
	.size	_ZN6Kalmar5indexILi2EEC1Ev, .Lfunc_end51-_ZN6Kalmar5indexILi2EEC1Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC2ERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEC2ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC2ERKS1_,@function
_ZN6Kalmar5indexILi2EEC2ERKS1_:         # @_ZN6Kalmar5indexILi2EEC2ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end52:
	.size	_ZN6Kalmar5indexILi2EEC2ERKS1_, .Lfunc_end52-_ZN6Kalmar5indexILi2EEC2ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rax
	movq	%rdi, -24(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-16(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	movl	%esi, -28(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %edx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	-28(%rbp), %esi         # 4-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end53:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_, .Lfunc_end53-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2ERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC1ERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEC1ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC1ERKS1_,@function
_ZN6Kalmar5indexILi2EEC1ERKS1_:         # @_ZN6Kalmar5indexILi2EEC1ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi2EEC2ERKS1_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end54:
	.size	_ZN6Kalmar5indexILi2EEC1ERKS1_, .Lfunc_end54-_ZN6Kalmar5indexILi2EEC1ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC2Ei # -- Begin function _ZN6Kalmar5indexILi2EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC2Ei,@function
_ZN6Kalmar5indexILi2EEC2Ei:             # @_ZN6Kalmar5indexILi2EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end55:
	.size	_ZN6Kalmar5indexILi2EEC2Ei, .Lfunc_end55-_ZN6Kalmar5indexILi2EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end56:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei, .Lfunc_end56-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC1Ei # -- Begin function _ZN6Kalmar5indexILi2EEC1Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC1Ei,@function
_ZN6Kalmar5indexILi2EEC1Ei:             # @_ZN6Kalmar5indexILi2EEC1Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar5indexILi2EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end57:
	.size	_ZN6Kalmar5indexILi2EEC1Ei, .Lfunc_end57-_ZN6Kalmar5indexILi2EEC1Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC2EPKi # -- Begin function _ZN6Kalmar5indexILi2EEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC2EPKi,@function
_ZN6Kalmar5indexILi2EEC2EPKi:           # @_ZN6Kalmar5indexILi2EEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end58:
	.size	_ZN6Kalmar5indexILi2EEC2EPKi, .Lfunc_end58-_ZN6Kalmar5indexILi2EEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movl	(%rdx), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	movl	4(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end59:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi, .Lfunc_end59-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC1EPKi # -- Begin function _ZN6Kalmar5indexILi2EEC1EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC1EPKi,@function
_ZN6Kalmar5indexILi2EEC1EPKi:           # @_ZN6Kalmar5indexILi2EEC1EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi2EEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end60:
	.size	_ZN6Kalmar5indexILi2EEC1EPKi, .Lfunc_end60-_ZN6Kalmar5indexILi2EEC1EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC2EPi # -- Begin function _ZN6Kalmar5indexILi2EEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC2EPi,@function
_ZN6Kalmar5indexILi2EEC2EPi:            # @_ZN6Kalmar5indexILi2EEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end61:
	.size	_ZN6Kalmar5indexILi2EEC2EPi, .Lfunc_end61-_ZN6Kalmar5indexILi2EEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movl	(%rdx), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	movl	4(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end62:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi, .Lfunc_end62-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEC1EPi # -- Begin function _ZN6Kalmar5indexILi2EEC1EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEC1EPi,@function
_ZN6Kalmar5indexILi2EEC1EPi:            # @_ZN6Kalmar5indexILi2EEC1EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi2EEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end63:
	.size	_ZN6Kalmar5indexILi2EEC1EPi, .Lfunc_end63-_ZN6Kalmar5indexILi2EEC1EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEaSERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEaSERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEaSERKS1_,@function
_ZN6Kalmar5indexILi2EEaSERKS1_:         # @_ZN6Kalmar5indexILi2EEaSERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end64:
	.size	_ZN6Kalmar5indexILi2EEaSERKS1_, .Lfunc_end64-_ZN6Kalmar5indexILi2EEaSERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEaSEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEaSEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end65:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_, .Lfunc_end65-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi2EEixEj # -- Begin function _ZNK6Kalmar5indexILi2EEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi2EEixEj,@function
_ZNK6Kalmar5indexILi2EEixEj:            # @_ZNK6Kalmar5indexILi2EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end66:
	.size	_ZNK6Kalmar5indexILi2EEixEj, .Lfunc_end66-_ZNK6Kalmar5indexILi2EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj # -- Begin function _ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj,@function
_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj: # @_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end67:
	.size	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj, .Lfunc_end67-_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEixEj # -- Begin function _ZN6Kalmar5indexILi2EEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEixEj,@function
_ZN6Kalmar5indexILi2EEixEj:             # @_ZN6Kalmar5indexILi2EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end68:
	.size	_ZN6Kalmar5indexILi2EEixEj, .Lfunc_end68-_ZN6Kalmar5indexILi2EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EE3getEv
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end69:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj, .Lfunc_end69-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi2EEeqERKS1_ # -- Begin function _ZNK6Kalmar5indexILi2EEeqERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi2EEeqERKS1_,@function
_ZNK6Kalmar5indexILi2EEeqERKS1_:        # @_ZNK6Kalmar5indexILi2EEeqERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end70:
	.size	_ZNK6Kalmar5indexILi2EEeqERKS1_, .Lfunc_end70-_ZNK6Kalmar5indexILi2EEeqERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	$1, %esi
	callq	_ZNK6Kalmar5indexILi2EEixEj
	movq	-16(%rbp), %rdi
	movl	$1, %esi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi2EEixEj
	xorl	%ecx, %ecx
                                        # kill: def $cl killed $cl killed $ecx
	movl	-20(%rbp), %edx         # 4-byte Reload
	cmpl	%eax, %edx
	movb	%cl, -21(%rbp)          # 1-byte Spill
	jne	.LBB71_2
# %bb.1:                                # %land.rhs
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_
	movb	%al, -21(%rbp)          # 1-byte Spill
.LBB71_2:                               # %land.end
	movb	-21(%rbp), %al          # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end71:
	.size	_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_, .Lfunc_end71-_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi2EEneERKS1_ # -- Begin function _ZNK6Kalmar5indexILi2EEneERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi2EEneERKS1_,@function
_ZNK6Kalmar5indexILi2EEneERKS1_:        # @_ZNK6Kalmar5indexILi2EEneERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZNK6Kalmar5indexILi2EEeqERKS1_
	xorb	$-1, %al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end72:
	.size	_ZNK6Kalmar5indexILi2EEneERKS1_, .Lfunc_end72-_ZNK6Kalmar5indexILi2EEneERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEpLERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEpLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEpLERKS1_,@function
_ZN6Kalmar5indexILi2EEpLERKS1_:         # @_ZN6Kalmar5indexILi2EEpLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end73:
	.size	_ZN6Kalmar5indexILi2EEpLERKS1_, .Lfunc_end73-_ZN6Kalmar5indexILi2EEpLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEpLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end74:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_, .Lfunc_end74-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmIERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEmIERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmIERKS1_,@function
_ZN6Kalmar5indexILi2EEmIERKS1_:         # @_ZN6Kalmar5indexILi2EEmIERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end75:
	.size	_ZN6Kalmar5indexILi2EEmIERKS1_, .Lfunc_end75-_ZN6Kalmar5indexILi2EEmIERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEmIEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end76:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_, .Lfunc_end76-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmLERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEmLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmLERKS1_,@function
_ZN6Kalmar5indexILi2EEmLERKS1_:         # @_ZN6Kalmar5indexILi2EEmLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end77:
	.size	_ZN6Kalmar5indexILi2EEmLERKS1_, .Lfunc_end77-_ZN6Kalmar5indexILi2EEmLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEmLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end78:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_, .Lfunc_end78-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEdVERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EEdVERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEdVERKS1_,@function
_ZN6Kalmar5indexILi2EEdVERKS1_:         # @_ZN6Kalmar5indexILi2EEdVERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end79:
	.size	_ZN6Kalmar5indexILi2EEdVERKS1_, .Lfunc_end79-_ZN6Kalmar5indexILi2EEdVERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEdVEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end80:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_, .Lfunc_end80-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EErMERKS1_ # -- Begin function _ZN6Kalmar5indexILi2EErMERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EErMERKS1_,@function
_ZN6Kalmar5indexILi2EErMERKS1_:         # @_ZN6Kalmar5indexILi2EErMERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end81:
	.size	_ZN6Kalmar5indexILi2EErMERKS1_, .Lfunc_end81-_ZN6Kalmar5indexILi2EErMERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EErMEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end82:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_, .Lfunc_end82-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEpLEi # -- Begin function _ZN6Kalmar5indexILi2EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEpLEi,@function
_ZN6Kalmar5indexILi2EEpLEi:             # @_ZN6Kalmar5indexILi2EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end83:
	.size	_ZN6Kalmar5indexILi2EEpLEi, .Lfunc_end83-_ZN6Kalmar5indexILi2EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEpLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end84:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi, .Lfunc_end84-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmIEi # -- Begin function _ZN6Kalmar5indexILi2EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmIEi,@function
_ZN6Kalmar5indexILi2EEmIEi:             # @_ZN6Kalmar5indexILi2EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end85:
	.size	_ZN6Kalmar5indexILi2EEmIEi, .Lfunc_end85-_ZN6Kalmar5indexILi2EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEmIEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end86:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi, .Lfunc_end86-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmLEi # -- Begin function _ZN6Kalmar5indexILi2EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmLEi,@function
_ZN6Kalmar5indexILi2EEmLEi:             # @_ZN6Kalmar5indexILi2EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end87:
	.size	_ZN6Kalmar5indexILi2EEmLEi, .Lfunc_end87-_ZN6Kalmar5indexILi2EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEmLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end88:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi, .Lfunc_end88-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEdVEi # -- Begin function _ZN6Kalmar5indexILi2EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEdVEi,@function
_ZN6Kalmar5indexILi2EEdVEi:             # @_ZN6Kalmar5indexILi2EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end89:
	.size	_ZN6Kalmar5indexILi2EEdVEi, .Lfunc_end89-_ZN6Kalmar5indexILi2EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEdVEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end90:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi, .Lfunc_end90-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EErMEi # -- Begin function _ZN6Kalmar5indexILi2EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EErMEi,@function
_ZN6Kalmar5indexILi2EErMEi:             # @_ZN6Kalmar5indexILi2EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end91:
	.size	_ZN6Kalmar5indexILi2EErMEi, .Lfunc_end91-_ZN6Kalmar5indexILi2EErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EErMEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	movq	-40(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end92:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi, .Lfunc_end92-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEppEv # -- Begin function _ZN6Kalmar5indexILi2EEppEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEppEv,@function
_ZN6Kalmar5indexILi2EEppEv:             # @_ZN6Kalmar5indexILi2EEppEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end93:
	.size	_ZN6Kalmar5indexILi2EEppEv, .Lfunc_end93-_ZN6Kalmar5indexILi2EEppEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEppEi # -- Begin function _ZN6Kalmar5indexILi2EEppEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEppEi,@function
_ZN6Kalmar5indexILi2EEppEi:             # @_ZN6Kalmar5indexILi2EEppEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi2EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end94:
	.size	_ZN6Kalmar5indexILi2EEppEi, .Lfunc_end94-_ZN6Kalmar5indexILi2EEppEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmmEv # -- Begin function _ZN6Kalmar5indexILi2EEmmEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmmEv,@function
_ZN6Kalmar5indexILi2EEmmEv:             # @_ZN6Kalmar5indexILi2EEmmEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end95:
	.size	_ZN6Kalmar5indexILi2EEmmEv, .Lfunc_end95-_ZN6Kalmar5indexILi2EEmmEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi2EEmmEi # -- Begin function _ZN6Kalmar5indexILi2EEmmEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi2EEmmEi,@function
_ZN6Kalmar5indexILi2EEmmEi:             # @_ZN6Kalmar5indexILi2EEmmEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi2EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end96:
	.size	_ZN6Kalmar5indexILi2EEmmEi, .Lfunc_end96-_ZN6Kalmar5indexILi2EEmmEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC2Ev # -- Begin function _ZN6Kalmar5indexILi3EEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC2Ev,@function
_ZN6Kalmar5indexILi3EEC2Ev:             # @_ZN6Kalmar5indexILi3EEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end97:
	.size	_ZN6Kalmar5indexILi3EEC2Ev, .Lfunc_end97-_ZN6Kalmar5indexILi3EEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	xorl	%esi, %esi
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	%rcx, %rdi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	xorl	%esi, %esi
	movq	-16(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	xorl	%esi, %esi
	movq	-16(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end98:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev, .Lfunc_end98-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC1Ev # -- Begin function _ZN6Kalmar5indexILi3EEC1Ev
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC1Ev,@function
_ZN6Kalmar5indexILi3EEC1Ev:             # @_ZN6Kalmar5indexILi3EEC1Ev
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rdi
	callq	_ZN6Kalmar5indexILi3EEC2Ev
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end99:
	.size	_ZN6Kalmar5indexILi3EEC1Ev, .Lfunc_end99-_ZN6Kalmar5indexILi3EEC1Ev
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC2ERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEC2ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC2ERKS1_,@function
_ZN6Kalmar5indexILi3EEC2ERKS1_:         # @_ZN6Kalmar5indexILi3EEC2ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end100:
	.size	_ZN6Kalmar5indexILi3EEC2ERKS1_, .Lfunc_end100-_ZN6Kalmar5indexILi3EEC2ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rax
	movq	%rdi, -24(%rbp)         # 8-byte Spill
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-16(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	movl	%esi, -28(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %edx
	movq	-16(%rbp), %rax
	addq	$16, %rax
	movq	%rax, %rdi
	movl	%edx, -32(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	-28(%rbp), %esi         # 4-byte Reload
	movl	-32(%rbp), %edx         # 4-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end101:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_, .Lfunc_end101-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2ERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC1ERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEC1ERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC1ERKS1_,@function
_ZN6Kalmar5indexILi3EEC1ERKS1_:         # @_ZN6Kalmar5indexILi3EEC1ERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi3EEC2ERKS1_
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end102:
	.size	_ZN6Kalmar5indexILi3EEC1ERKS1_, .Lfunc_end102-_ZN6Kalmar5indexILi3EEC1ERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC2Ei # -- Begin function _ZN6Kalmar5indexILi3EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC2Ei,@function
_ZN6Kalmar5indexILi3EEC2Ei:             # @_ZN6Kalmar5indexILi3EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end103:
	.size	_ZN6Kalmar5indexILi3EEC2Ei, .Lfunc_end103-_ZN6Kalmar5indexILi3EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end104:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei, .Lfunc_end104-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC1Ei # -- Begin function _ZN6Kalmar5indexILi3EEC1Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC1Ei,@function
_ZN6Kalmar5indexILi3EEC1Ei:             # @_ZN6Kalmar5indexILi3EEC1Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar5indexILi3EEC2Ei
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end105:
	.size	_ZN6Kalmar5indexILi3EEC1Ei, .Lfunc_end105-_ZN6Kalmar5indexILi3EEC1Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC2EPKi # -- Begin function _ZN6Kalmar5indexILi3EEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC2EPKi,@function
_ZN6Kalmar5indexILi3EEC2EPKi:           # @_ZN6Kalmar5indexILi3EEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end106:
	.size	_ZN6Kalmar5indexILi3EEC2EPKi, .Lfunc_end106-_ZN6Kalmar5indexILi3EEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movl	(%rdx), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	movl	4(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	movl	8(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end107:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi, .Lfunc_end107-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC1EPKi # -- Begin function _ZN6Kalmar5indexILi3EEC1EPKi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC1EPKi,@function
_ZN6Kalmar5indexILi3EEC1EPKi:           # @_ZN6Kalmar5indexILi3EEC1EPKi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi3EEC2EPKi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end108:
	.size	_ZN6Kalmar5indexILi3EEC1EPKi, .Lfunc_end108-_ZN6Kalmar5indexILi3EEC1EPKi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC2EPi # -- Begin function _ZN6Kalmar5indexILi3EEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC2EPi,@function
_ZN6Kalmar5indexILi3EEC2EPi:            # @_ZN6Kalmar5indexILi3EEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end109:
	.size	_ZN6Kalmar5indexILi3EEC2EPi, .Lfunc_end109-_ZN6Kalmar5indexILi3EEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movl	(%rdx), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	movl	4(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	movl	8(%rcx), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end110:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi, .Lfunc_end110-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEC1EPi # -- Begin function _ZN6Kalmar5indexILi3EEC1EPi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEC1EPi,@function
_ZN6Kalmar5indexILi3EEC1EPi:            # @_ZN6Kalmar5indexILi3EEC1EPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar5indexILi3EEC2EPi
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end111:
	.size	_ZN6Kalmar5indexILi3EEC1EPi, .Lfunc_end111-_ZN6Kalmar5indexILi3EEC1EPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEaSERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEaSERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEaSERKS1_,@function
_ZN6Kalmar5indexILi3EEaSERKS1_:         # @_ZN6Kalmar5indexILi3EEaSERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end112:
	.size	_ZN6Kalmar5indexILi3EEaSERKS1_, .Lfunc_end112-_ZN6Kalmar5indexILi3EEaSERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEaSEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEaSEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EEaSEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end113:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_, .Lfunc_end113-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi3EEixEj # -- Begin function _ZNK6Kalmar5indexILi3EEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi3EEixEj,@function
_ZNK6Kalmar5indexILi3EEixEj:            # @_ZNK6Kalmar5indexILi3EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end114:
	.size	_ZNK6Kalmar5indexILi3EEixEj, .Lfunc_end114-_ZNK6Kalmar5indexILi3EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj # -- Begin function _ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.p2align	4, 0x90
	.type	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj,@function
_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj: # @_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end115:
	.size	_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj, .Lfunc_end115-_ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEixEj # -- Begin function _ZN6Kalmar5indexILi3EEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEixEj,@function
_ZN6Kalmar5indexILi3EEixEj:             # @_ZN6Kalmar5indexILi3EEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rdi
	movl	-12(%rbp), %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end116:
	.size	_ZN6Kalmar5indexILi3EEixEj, .Lfunc_end116-_ZN6Kalmar5indexILi3EEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, %edx
	shlq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi0EE3getEv
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end117:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj, .Lfunc_end117-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi3EEeqERKS1_ # -- Begin function _ZNK6Kalmar5indexILi3EEeqERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi3EEeqERKS1_,@function
_ZNK6Kalmar5indexILi3EEeqERKS1_:        # @_ZNK6Kalmar5indexILi3EEeqERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end118:
	.size	_ZNK6Kalmar5indexILi3EEeqERKS1_, .Lfunc_end118-_ZNK6Kalmar5indexILi3EEeqERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	$2, %esi
	callq	_ZNK6Kalmar5indexILi3EEixEj
	movq	-16(%rbp), %rdi
	movl	$2, %esi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi3EEixEj
	xorl	%ecx, %ecx
                                        # kill: def $cl killed $cl killed $ecx
	movl	-20(%rbp), %edx         # 4-byte Reload
	cmpl	%eax, %edx
	movb	%cl, -21(%rbp)          # 1-byte Spill
	jne	.LBB119_2
# %bb.1:                                # %land.rhs
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_
	movb	%al, -21(%rbp)          # 1-byte Spill
.LBB119_2:                              # %land.end
	movb	-21(%rbp), %al          # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end119:
	.size	_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_, .Lfunc_end119-_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar5indexILi3EEneERKS1_ # -- Begin function _ZNK6Kalmar5indexILi3EEneERKS1_
	.p2align	4, 0x90
	.type	_ZNK6Kalmar5indexILi3EEneERKS1_,@function
_ZNK6Kalmar5indexILi3EEneERKS1_:        # @_ZNK6Kalmar5indexILi3EEneERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZNK6Kalmar5indexILi3EEeqERKS1_
	xorb	$-1, %al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end120:
	.size	_ZNK6Kalmar5indexILi3EEneERKS1_, .Lfunc_end120-_ZNK6Kalmar5indexILi3EEneERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEpLERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEpLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEpLERKS1_,@function
_ZN6Kalmar5indexILi3EEpLERKS1_:         # @_ZN6Kalmar5indexILi3EEpLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end121:
	.size	_ZN6Kalmar5indexILi3EEpLERKS1_, .Lfunc_end121-_ZN6Kalmar5indexILi3EEpLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEpLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EEpLEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end122:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_, .Lfunc_end122-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmIERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEmIERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmIERKS1_,@function
_ZN6Kalmar5indexILi3EEmIERKS1_:         # @_ZN6Kalmar5indexILi3EEmIERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end123:
	.size	_ZN6Kalmar5indexILi3EEmIERKS1_, .Lfunc_end123-_ZN6Kalmar5indexILi3EEmIERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEmIEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EEmIEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end124:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_, .Lfunc_end124-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmLERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEmLERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmLERKS1_,@function
_ZN6Kalmar5indexILi3EEmLERKS1_:         # @_ZN6Kalmar5indexILi3EEmLERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end125:
	.size	_ZN6Kalmar5indexILi3EEmLERKS1_, .Lfunc_end125-_ZN6Kalmar5indexILi3EEmLERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEmLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EEmLEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end126:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_, .Lfunc_end126-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEdVERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EEdVERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEdVERKS1_,@function
_ZN6Kalmar5indexILi3EEdVERKS1_:         # @_ZN6Kalmar5indexILi3EEdVERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end127:
	.size	_ZN6Kalmar5indexILi3EEdVERKS1_, .Lfunc_end127-_ZN6Kalmar5indexILi3EEdVERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EEdVEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EEdVEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end128:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_, .Lfunc_end128-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EErMERKS1_ # -- Begin function _ZN6Kalmar5indexILi3EErMERKS1_
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EErMERKS1_,@function
_ZN6Kalmar5indexILi3EErMERKS1_:         # @_ZN6Kalmar5indexILi3EErMERKS1_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rsi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end129:
	.size	_ZN6Kalmar5indexILi3EErMERKS1_, .Lfunc_end129-_ZN6Kalmar5indexILi3EErMERKS1_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, -56(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi0EE3getEv
	movl	(%rax), %esi
	movq	-56(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
	movq	%rcx, %rdi
	movq	%rax, -64(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi1EE3getEv
	movl	(%rax), %esi
	movq	-64(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi1EErMEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movq	-16(%rbp), %rcx
	addq	$16, %rcx
	movq	%rcx, %rdi
	movq	%rax, -72(%rbp)         # 8-byte Spill
	callq	_ZNK6Kalmar12__index_leafILi2EE3getEv
	movl	(%rax), %esi
	movq	-72(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar12__index_leafILi2EErMEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end130:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_, .Lfunc_end130-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEpLEi # -- Begin function _ZN6Kalmar5indexILi3EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEpLEi,@function
_ZN6Kalmar5indexILi3EEpLEi:             # @_ZN6Kalmar5indexILi3EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end131:
	.size	_ZN6Kalmar5indexILi3EEpLEi, .Lfunc_end131-_ZN6Kalmar5indexILi3EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEpLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEpLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEpLEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end132:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi, .Lfunc_end132-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmIEi # -- Begin function _ZN6Kalmar5indexILi3EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmIEi,@function
_ZN6Kalmar5indexILi3EEmIEi:             # @_ZN6Kalmar5indexILi3EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end133:
	.size	_ZN6Kalmar5indexILi3EEmIEi, .Lfunc_end133-_ZN6Kalmar5indexILi3EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmIEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEmIEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEmIEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end134:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi, .Lfunc_end134-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmLEi # -- Begin function _ZN6Kalmar5indexILi3EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmLEi,@function
_ZN6Kalmar5indexILi3EEmLEi:             # @_ZN6Kalmar5indexILi3EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end135:
	.size	_ZN6Kalmar5indexILi3EEmLEi, .Lfunc_end135-_ZN6Kalmar5indexILi3EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEmLEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEmLEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEmLEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end136:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi, .Lfunc_end136-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEdVEi # -- Begin function _ZN6Kalmar5indexILi3EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEdVEi,@function
_ZN6Kalmar5indexILi3EEdVEi:             # @_ZN6Kalmar5indexILi3EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end137:
	.size	_ZN6Kalmar5indexILi3EEdVEi, .Lfunc_end137-_ZN6Kalmar5indexILi3EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEdVEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEdVEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEdVEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end138:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi, .Lfunc_end138-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EErMEi # -- Begin function _ZN6Kalmar5indexILi3EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EErMEi,@function
_ZN6Kalmar5indexILi3EErMEi:             # @_ZN6Kalmar5indexILi3EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi
	movq	-24(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end139:
	.size	_ZN6Kalmar5indexILi3EErMEi, .Lfunc_end139-_ZN6Kalmar5indexILi3EErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -48(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EErMEi
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EErMEi
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-12(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EErMEi
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rsi
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-48(%rbp), %rdi         # 8-byte Reload
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end140:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi, .Lfunc_end140-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEppEv # -- Begin function _ZN6Kalmar5indexILi3EEppEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEppEv,@function
_ZN6Kalmar5indexILi3EEppEv:             # @_ZN6Kalmar5indexILi3EEppEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end141:
	.size	_ZN6Kalmar5indexILi3EEppEv, .Lfunc_end141-_ZN6Kalmar5indexILi3EEppEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEppEi # -- Begin function _ZN6Kalmar5indexILi3EEppEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEppEi,@function
_ZN6Kalmar5indexILi3EEppEi:             # @_ZN6Kalmar5indexILi3EEppEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi3EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end142:
	.size	_ZN6Kalmar5indexILi3EEppEi, .Lfunc_end142-_ZN6Kalmar5indexILi3EEppEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmmEv # -- Begin function _ZN6Kalmar5indexILi3EEmmEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmmEv,@function
_ZN6Kalmar5indexILi3EEmmEv:             # @_ZN6Kalmar5indexILi3EEmmEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	$1, %esi
	movq	%rax, -16(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	movq	-16(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end143:
	.size	_ZN6Kalmar5indexILi3EEmmEv, .Lfunc_end143-_ZN6Kalmar5indexILi3EEmmEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar5indexILi3EEmmEi # -- Begin function _ZN6Kalmar5indexILi3EEmmEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar5indexILi3EEmmEi,@function
_ZN6Kalmar5indexILi3EEmmEi:             # @_ZN6Kalmar5indexILi3EEmmEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, %rax
	movq	%rdi, %rcx
	movq	%rcx, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	%edx, -20(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	movq	%rcx, -40(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar5indexILi3EEC1ERKS1_
	movq	-40(%rbp), %rdi         # 8-byte Reload
	movl	$1, %esi
	callq	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	movq	-32(%rbp), %rcx         # 8-byte Reload
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	%rcx, %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end144:
	.size	_ZN6Kalmar5indexILi3EEmmEi, .Lfunc_end144-_ZN6Kalmar5indexILi3EEmmEi
	.cfi_endproc
                                        # -- End function
	.globl	_Z10HelloWorldPi        # -- Begin function _Z10HelloWorldPi
	.p2align	4, 0x90
	.type	_Z10HelloWorldPi,@function
_Z10HelloWorldPi:                       # @_Z10HelloWorldPi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end145:
	.size	_Z10HelloWorldPi, .Lfunc_end145-_Z10HelloWorldPi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEC2Ei # -- Begin function _ZN6Kalmar12__index_leafILi0EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEC2Ei,@function
_ZN6Kalmar12__index_leafILi0EEC2Ei:     # @_ZN6Kalmar12__index_leafILi0EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end146:
	.size	_ZN6Kalmar12__index_leafILi0EEC2Ei, .Lfunc_end146-_ZN6Kalmar12__index_leafILi0EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar12__index_leafILi0EE3getEv # -- Begin function _ZNK6Kalmar12__index_leafILi0EE3getEv
	.p2align	4, 0x90
	.type	_ZNK6Kalmar12__index_leafILi0EE3getEv,@function
_ZNK6Kalmar12__index_leafILi0EE3getEv:  # @_ZNK6Kalmar12__index_leafILi0EE3getEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end147:
	.size	_ZNK6Kalmar12__index_leafILi0EE3getEv, .Lfunc_end147-_ZNK6Kalmar12__index_leafILi0EE3getEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, -8(%rbp)
	movq	%rdi, -16(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end148:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_, .Lfunc_end148-_ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEaSEi # -- Begin function _ZN6Kalmar12__index_leafILi0EEaSEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEaSEi,@function
_ZN6Kalmar12__index_leafILi0EEaSEi:     # @_ZN6Kalmar12__index_leafILi0EEaSEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end149:
	.size	_ZN6Kalmar12__index_leafILi0EEaSEi, .Lfunc_end149-_ZN6Kalmar12__index_leafILi0EEaSEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EE3getEv # -- Begin function _ZN6Kalmar12__index_leafILi0EE3getEv
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EE3getEv,@function
_ZN6Kalmar12__index_leafILi0EE3getEv:   # @_ZN6Kalmar12__index_leafILi0EE3getEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end150:
	.size	_ZN6Kalmar12__index_leafILi0EE3getEv, .Lfunc_end150-_ZN6Kalmar12__index_leafILi0EE3getEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEpLEi # -- Begin function _ZN6Kalmar12__index_leafILi0EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEpLEi,@function
_ZN6Kalmar12__index_leafILi0EEpLEi:     # @_ZN6Kalmar12__index_leafILi0EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	addl	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end151:
	.size	_ZN6Kalmar12__index_leafILi0EEpLEi, .Lfunc_end151-_ZN6Kalmar12__index_leafILi0EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEmIEi # -- Begin function _ZN6Kalmar12__index_leafILi0EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEmIEi,@function
_ZN6Kalmar12__index_leafILi0EEmIEi:     # @_ZN6Kalmar12__index_leafILi0EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	subl	%ecx, %edx
	movl	%edx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end152:
	.size	_ZN6Kalmar12__index_leafILi0EEmIEi, .Lfunc_end152-_ZN6Kalmar12__index_leafILi0EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEmLEi # -- Begin function _ZN6Kalmar12__index_leafILi0EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEmLEi,@function
_ZN6Kalmar12__index_leafILi0EEmLEi:     # @_ZN6Kalmar12__index_leafILi0EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	imull	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end153:
	.size	_ZN6Kalmar12__index_leafILi0EEmLEi, .Lfunc_end153-_ZN6Kalmar12__index_leafILi0EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EEdVEi # -- Begin function _ZN6Kalmar12__index_leafILi0EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EEdVEi,@function
_ZN6Kalmar12__index_leafILi0EEdVEi:     # @_ZN6Kalmar12__index_leafILi0EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%eax, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end154:
	.size	_ZN6Kalmar12__index_leafILi0EEdVEi, .Lfunc_end154-_ZN6Kalmar12__index_leafILi0EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi0EErMEi # -- Begin function _ZN6Kalmar12__index_leafILi0EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi0EErMEi,@function
_ZN6Kalmar12__index_leafILi0EErMEi:     # @_ZN6Kalmar12__index_leafILi0EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%edx, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end155:
	.size	_ZN6Kalmar12__index_leafILi0EErMEi, .Lfunc_end155-_ZN6Kalmar12__index_leafILi0EErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEC2Ei # -- Begin function _ZN6Kalmar12__index_leafILi1EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEC2Ei,@function
_ZN6Kalmar12__index_leafILi1EEC2Ei:     # @_ZN6Kalmar12__index_leafILi1EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end156:
	.size	_ZN6Kalmar12__index_leafILi1EEC2Ei, .Lfunc_end156-_ZN6Kalmar12__index_leafILi1EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar12__index_leafILi1EE3getEv # -- Begin function _ZNK6Kalmar12__index_leafILi1EE3getEv
	.p2align	4, 0x90
	.type	_ZNK6Kalmar12__index_leafILi1EE3getEv,@function
_ZNK6Kalmar12__index_leafILi1EE3getEv:  # @_ZNK6Kalmar12__index_leafILi1EE3getEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end157:
	.size	_ZNK6Kalmar12__index_leafILi1EE3getEv, .Lfunc_end157-_ZNK6Kalmar12__index_leafILi1EE3getEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rcx
	movl	-12(%rbp), %esi
	movq	%rcx, %rdi
	movq	%rax, -24(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-24(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-16(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end158:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_, .Lfunc_end158-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEC2IJiiEEEDpT_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rdi, -24(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end159:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_, .Lfunc_end159-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEaSEi # -- Begin function _ZN6Kalmar12__index_leafILi1EEaSEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEaSEi,@function
_ZN6Kalmar12__index_leafILi1EEaSEi:     # @_ZN6Kalmar12__index_leafILi1EEaSEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end160:
	.size	_ZN6Kalmar12__index_leafILi1EEaSEi, .Lfunc_end160-_ZN6Kalmar12__index_leafILi1EEaSEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	xorl	%eax, %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	%eax, %esi
	callq	_ZNK6Kalmar5indexILi2EEixEj
	xorl	%esi, %esi
	movq	-16(%rbp), %rdi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi2EEixEj
	movl	-20(%rbp), %ecx         # 4-byte Reload
	cmpl	%eax, %ecx
	sete	%dl
	andb	$1, %dl
	movzbl	%dl, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end161:
	.size	_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_, .Lfunc_end161-_ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEpLEi # -- Begin function _ZN6Kalmar12__index_leafILi1EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEpLEi,@function
_ZN6Kalmar12__index_leafILi1EEpLEi:     # @_ZN6Kalmar12__index_leafILi1EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	addl	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end162:
	.size	_ZN6Kalmar12__index_leafILi1EEpLEi, .Lfunc_end162-_ZN6Kalmar12__index_leafILi1EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEmIEi # -- Begin function _ZN6Kalmar12__index_leafILi1EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEmIEi,@function
_ZN6Kalmar12__index_leafILi1EEmIEi:     # @_ZN6Kalmar12__index_leafILi1EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	subl	%ecx, %edx
	movl	%edx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end163:
	.size	_ZN6Kalmar12__index_leafILi1EEmIEi, .Lfunc_end163-_ZN6Kalmar12__index_leafILi1EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEmLEi # -- Begin function _ZN6Kalmar12__index_leafILi1EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEmLEi,@function
_ZN6Kalmar12__index_leafILi1EEmLEi:     # @_ZN6Kalmar12__index_leafILi1EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	imull	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end164:
	.size	_ZN6Kalmar12__index_leafILi1EEmLEi, .Lfunc_end164-_ZN6Kalmar12__index_leafILi1EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EEdVEi # -- Begin function _ZN6Kalmar12__index_leafILi1EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EEdVEi,@function
_ZN6Kalmar12__index_leafILi1EEdVEi:     # @_ZN6Kalmar12__index_leafILi1EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%eax, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end165:
	.size	_ZN6Kalmar12__index_leafILi1EEdVEi, .Lfunc_end165-_ZN6Kalmar12__index_leafILi1EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi1EErMEi # -- Begin function _ZN6Kalmar12__index_leafILi1EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi1EErMEi,@function
_ZN6Kalmar12__index_leafILi1EErMEi:     # @_ZN6Kalmar12__index_leafILi1EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%edx, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end166:
	.size	_ZN6Kalmar12__index_leafILi1EErMEi, .Lfunc_end166-_ZN6Kalmar12__index_leafILi1EErMEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEC2Ei # -- Begin function _ZN6Kalmar12__index_leafILi2EEC2Ei
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEC2Ei,@function
_ZN6Kalmar12__index_leafILi2EEC2Ei:     # @_ZN6Kalmar12__index_leafILi2EEC2Ei
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end167:
	.size	_ZN6Kalmar12__index_leafILi2EEC2Ei, .Lfunc_end167-_ZN6Kalmar12__index_leafILi2EEC2Ei
	.cfi_endproc
                                        # -- End function
	.weak	_ZNK6Kalmar12__index_leafILi2EE3getEv # -- Begin function _ZNK6Kalmar12__index_leafILi2EE3getEv
	.p2align	4, 0x90
	.type	_ZNK6Kalmar12__index_leafILi2EE3getEv,@function
_ZNK6Kalmar12__index_leafILi2EE3getEv:  # @_ZNK6Kalmar12__index_leafILi2EE3getEv
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end168:
	.size	_ZNK6Kalmar12__index_leafILi2EE3getEv, .Lfunc_end168-_ZNK6Kalmar12__index_leafILi2EE3getEv
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	movl	-12(%rbp), %esi
	movq	%rax, -32(%rbp)         # 8-byte Spill
	callq	_ZN6Kalmar12__index_leafILi0EEC2Ei
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$8, %rax
	movl	-16(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi1EEC2Ei
	movq	-32(%rbp), %rax         # 8-byte Reload
	addq	$16, %rax
	movl	-20(%rbp), %esi
	movq	%rax, %rdi
	callq	_ZN6Kalmar12__index_leafILi2EEC2Ei
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end169:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_, .Lfunc_end169-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEC2IJiiiEEEDpT_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_ # -- Begin function _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	.p2align	4, 0x90
	.type	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_,@function
_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_: # @_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rsi, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	%rdi, -32(%rbp)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end170:
	.size	_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_, .Lfunc_end170-_ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEaSEi # -- Begin function _ZN6Kalmar12__index_leafILi2EEaSEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEaSEi,@function
_ZN6Kalmar12__index_leafILi2EEaSEi:     # @_ZN6Kalmar12__index_leafILi2EEaSEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end171:
	.size	_ZN6Kalmar12__index_leafILi2EEaSEi, .Lfunc_end171-_ZN6Kalmar12__index_leafILi2EEaSEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	$1, %esi
	callq	_ZNK6Kalmar5indexILi3EEixEj
	movq	-16(%rbp), %rdi
	movl	$1, %esi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi3EEixEj
	xorl	%ecx, %ecx
                                        # kill: def $cl killed $cl killed $ecx
	movl	-20(%rbp), %edx         # 4-byte Reload
	cmpl	%eax, %edx
	movb	%cl, -21(%rbp)          # 1-byte Spill
	jne	.LBB172_2
# %bb.1:                                # %land.rhs
	movq	-8(%rbp), %rdi
	movq	-16(%rbp), %rsi
	callq	_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_
	movb	%al, -21(%rbp)          # 1-byte Spill
.LBB172_2:                              # %land.end
	movb	-21(%rbp), %al          # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end172:
	.size	_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_, .Lfunc_end172-_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_ # -- Begin function _ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_
	.p2align	4, 0x90
	.type	_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_,@function
_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_: # @_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	xorl	%eax, %eax
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdi
	movl	%eax, %esi
	callq	_ZNK6Kalmar5indexILi3EEixEj
	xorl	%esi, %esi
	movq	-16(%rbp), %rdi
	movl	%eax, -20(%rbp)         # 4-byte Spill
	callq	_ZNK6Kalmar5indexILi3EEixEj
	movl	-20(%rbp), %ecx         # 4-byte Reload
	cmpl	%eax, %ecx
	sete	%dl
	andb	$1, %dl
	movzbl	%dl, %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end173:
	.size	_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_, .Lfunc_end173-_ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEpLEi # -- Begin function _ZN6Kalmar12__index_leafILi2EEpLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEpLEi,@function
_ZN6Kalmar12__index_leafILi2EEpLEi:     # @_ZN6Kalmar12__index_leafILi2EEpLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	addl	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end174:
	.size	_ZN6Kalmar12__index_leafILi2EEpLEi, .Lfunc_end174-_ZN6Kalmar12__index_leafILi2EEpLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEmIEi # -- Begin function _ZN6Kalmar12__index_leafILi2EEmIEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEmIEi,@function
_ZN6Kalmar12__index_leafILi2EEmIEi:     # @_ZN6Kalmar12__index_leafILi2EEmIEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	subl	%ecx, %edx
	movl	%edx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end175:
	.size	_ZN6Kalmar12__index_leafILi2EEmIEi, .Lfunc_end175-_ZN6Kalmar12__index_leafILi2EEmIEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEmLEi # -- Begin function _ZN6Kalmar12__index_leafILi2EEmLEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEmLEi,@function
_ZN6Kalmar12__index_leafILi2EEmLEi:     # @_ZN6Kalmar12__index_leafILi2EEmLEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	imull	(%rax), %ecx
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end176:
	.size	_ZN6Kalmar12__index_leafILi2EEmLEi, .Lfunc_end176-_ZN6Kalmar12__index_leafILi2EEmLEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EEdVEi # -- Begin function _ZN6Kalmar12__index_leafILi2EEdVEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EEdVEi,@function
_ZN6Kalmar12__index_leafILi2EEdVEi:     # @_ZN6Kalmar12__index_leafILi2EEdVEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%eax, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end177:
	.size	_ZN6Kalmar12__index_leafILi2EEdVEi, .Lfunc_end177-_ZN6Kalmar12__index_leafILi2EEdVEi
	.cfi_endproc
                                        # -- End function
	.weak	_ZN6Kalmar12__index_leafILi2EErMEi # -- Begin function _ZN6Kalmar12__index_leafILi2EErMEi
	.p2align	4, 0x90
	.type	_ZN6Kalmar12__index_leafILi2EErMEi,@function
_ZN6Kalmar12__index_leafILi2EErMEi:     # @_ZN6Kalmar12__index_leafILi2EErMEi
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	(%rax), %edx
	movq	%rax, -24(%rbp)         # 8-byte Spill
	movl	%edx, %eax
	cltd
	idivl	%ecx
	movq	-24(%rbp), %rdi         # 8-byte Reload
	movl	%edx, (%rdi)
	movq	%rdi, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end178:
	.size	_ZN6Kalmar12__index_leafILi2EErMEi, .Lfunc_end178-_ZN6Kalmar12__index_leafILi2EErMEi
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function _GLOBAL__sub_I_gpukernel.cpp
	.type	_GLOBAL__sub_I_gpukernel.cpp,@function
_GLOBAL__sub_I_gpukernel.cpp:           # @_GLOBAL__sub_I_gpukernel.cpp
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	__cxx_global_var_init
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end179:
	.size	_GLOBAL__sub_I_gpukernel.cpp, .Lfunc_end179-_GLOBAL__sub_I_gpukernel.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object  # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_gpukernel.cpp

	.ident	"HCC clang version 10.0.0 (/data/jenkins_workspace/centos_pipeline_job_2.8/rocm-rel-2.8/rocm-2.8-13-20190920/centos/external/hcc-tot/clang 61f97b4ab7a5a36553c6fad7bf72601666127554) (/data/jenkins_workspace/centos_pipeline_job_2.8/rocm-rel-2.8/rocm-2.8-13-20190920/centos/external/hcc-tot/compiler b7f876231af7fdaf52e419088b8ba9e0c3a61845) (based on HCC 2.8.19356-94aa8710-61f97b4ab7-b7f876231af )"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __cxx_global_var_init
	.addrsig_sym __cxa_atexit
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEaSERKS3_
	.addrsig_sym _ZNK6Kalmar5indexILi1EEixEj
	.addrsig_sym _ZNK6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEixEj
	.addrsig_sym _ZNK6Kalmar5indexILi1EEeqERKS1_
	.addrsig_sym _ZN6Kalmar12index_helperILi1ENS_5indexILi1EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEpLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmIEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEmLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEEdVEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEErMEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEaSERKS3_
	.addrsig_sym _ZNK6Kalmar5indexILi2EEixEj
	.addrsig_sym _ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEixEj
	.addrsig_sym _ZNK6Kalmar5indexILi2EEeqERKS1_
	.addrsig_sym _ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEpLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmIEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEmLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEEdVEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEErMEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEaSERKS3_
	.addrsig_sym _ZNK6Kalmar5indexILi3EEixEj
	.addrsig_sym _ZNK6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEixEj
	.addrsig_sym _ZNK6Kalmar5indexILi3EEeqERKS1_
	.addrsig_sym _ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMERKS3_
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEpLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmIEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEmLEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEEdVEi
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEErMEi
	.addrsig_sym _Z10HelloWorldPi
	.addrsig_sym _ZNK6Kalmar12__index_leafILi0EE3getEv
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0EEEEE9__swallowIJNS_12__index_leafILi0EEEEEEvDpT_
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EEaSEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EE3getEv
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EEpLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EEmIEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EEmLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EEdVEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi0EErMEi
	.addrsig_sym _ZNK6Kalmar12__index_leafILi1EE3getEv
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEEEEEvDpT_
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EEaSEi
	.addrsig_sym _ZN6Kalmar12index_helperILi1ENS_5indexILi2EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EEpLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EEmIEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EEmLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EEdVEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi1EErMEi
	.addrsig_sym _ZNK6Kalmar12__index_leafILi2EE3getEv
	.addrsig_sym _ZN6Kalmar10index_implINS_9__indicesIJLi0ELi1ELi2EEEEE9__swallowIJNS_12__index_leafILi0EEENS5_ILi1EEENS5_ILi2EEEEEEvDpT_
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EEaSEi
	.addrsig_sym _ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar12index_helperILi1ENS_5indexILi3EEEE5equalERKS2_S5_
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EEpLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EEmIEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EEmLEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EEdVEi
	.addrsig_sym _ZN6Kalmar12__index_leafILi2EErMEi
	.addrsig_sym _GLOBAL__sub_I_gpukernel.cpp
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
