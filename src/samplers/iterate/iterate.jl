function codegen_iterate_basicmcjob(job::BasicMCJob)
  result::Expr

  if isa(job.sampler, ARS)
    result = codegen_iterate_ars(job)
  elseif isa(job.sampler, MH)
    result = codegen_iterate_mh(job)
  elseif isa(job.sampler, RAM)
    result = codegen_iterate_ram(job)
  elseif isa(job.sampler, HMC)
    result = codegen_iterate_hmc(job)
  elseif isa(job.sampler, MALA)
    result = codegen_iterate_mala(job)
  elseif isa(job.sampler, SMMALA)
    result = codegen_iterate_smmala(job)
  end

  result
end
