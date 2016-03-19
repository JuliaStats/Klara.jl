codegen(::Type{Val{:iterate}}, job::BasicMCJob) = codegen(Val{:iterate}, typeof(job.sampler), job)
