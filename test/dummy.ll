target datalayout = "e-p:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @foo() {
entry:
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @bar(i32 %a) #0 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int"}
!4 = !{!"int"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{}
