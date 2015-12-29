function codegen_iterate_basicmcjob(job::BasicMCJob, outopts::Dict, plain::Bool)
  result::Expr

  if isa(job.sampler, MH)
    result = codegen_iterate_mh(job, outopts, plain)
  elseif isa(job.sampler, MALA)
    result = codegen_iterate_mala(job, outopts, plain)
  end

  result
end
