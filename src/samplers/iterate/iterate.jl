function codegen_iterate_basicmcjob(job::BasicMCJob, outopts::Dict)
  result::Expr

  if isa(job.sampler, MH)
    result = codegen_iterate_mh(job, outopts)
  elseif isa(job.sampler, MALA)
    result = codegen_iterate_mala(job, outopts)
  end

  result
end
