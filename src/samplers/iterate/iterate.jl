function codegen_iterate_basicmcjob(job::BasicMCJob)
  result::Expr

  if isa(job.sampler, MH)
    result = codegen_iterate_mh(job)
  elseif isa(job.sampler, RAM)
    result = codegen_iterate_ram(job)
  elseif isa(job.sampler, HMC)
    result = codegen_iterate_hmc(job)
  elseif isa(job.sampler, MALA)
    result = codegen_iterate_mala(job)
  end

  result
end
