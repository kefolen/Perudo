"""
Monte Carlo Agent Parallel Processing

This module contains parallel processing components for the MonteCarloAgent including:
- Worker function for multiprocessing simulation chunks
- Worker pool management utilities
- Parallel evaluation logic with fallback to sequential mode
"""

import multiprocessing as mp
import random
import time


def worker_run_chunk(agent, obs, action, chunk_size, seed_offset):
    """Module-level worker function for parallel chunk processing."""
    return agent._run_chunk_simulations(obs, action, chunk_size, seed_offset)


class ParallelProcessingMixin:
    """
    Mixin class providing parallel processing capabilities for Monte Carlo agents.
    
    This mixin adds parallel evaluation methods that can be mixed into the main
    MonteCarloAgent class to enable multiprocessing support.
    """
    
    def _get_worker_pool(self):
        """Get or create worker pool for parallel processing."""
        if self._worker_pool is None and self.enable_parallel:
            try:
                self._worker_pool = mp.Pool(processes=self.num_workers)
            except Exception:
                # Fallback to sequential if pool creation fails
                self.enable_parallel = False
                self._worker_pool = None
        return self._worker_pool
    
    def _close_worker_pool(self):
        """Close worker pool if it exists."""
        if self._worker_pool is not None:
            try:
                self._worker_pool.close()
                self._worker_pool.join()
            except Exception:
                pass
            finally:
                self._worker_pool = None

    def _run_chunk_simulations(self, obs, action, chunk_size, seed_offset=0):
        """Worker function to run a chunk of simulations."""
        # Create local random generator with seed offset for reproducibility
        local_rng = random.Random(self.rng.getstate()[1][0] + seed_offset)
        
        total = 0.0
        individual_results = [] if self.variance_reduction else None
        
        for _ in range(chunk_size):
            # Use local RNG for determinization sampling
            orig_rng = self.rng
            self.rng = local_rng
            try:
                hands = self.sample_determinization(obs)
                result = self.simulate_from_determinization(hands, obs, action)
                total += result
                
                # Store individual result for variance reduction if enabled
                if individual_results is not None:
                    individual_results.append(result)
            finally:
                self.rng = orig_rng
        
        return total, chunk_size, individual_results

    def _evaluate_action_parallel(self, obs, action, best_so_far=None):
        """Parallel evaluation using worker pools."""
        pool = self._get_worker_pool()
        if pool is None:
            # Fallback to sequential if pool creation failed
            return self._evaluate_action_sequential(obs, action, best_so_far)

        sims = self.N
        chunk = self.chunk_size
        full_chunks = sims // chunk
        rem = sims % chunk

        total = 0.0
        runs_done = 0
        
        # For variance reduction, collect individual results
        simulation_results = [] if self.variance_reduction else None

        # early stop thresholds
        early_margin = self.early_stop_margin
        best_mean = None
        if best_so_far is not None:
            best_mean = best_so_far

        try:
            # Process full chunks in parallel
            if full_chunks > 0:
                # Prepare chunk tasks with seed offsets for reproducibility
                chunk_tasks = []
                for i in range(full_chunks):
                    seed_offset = i * chunk * 1000  # Large offset to avoid seed collisions
                    chunk_tasks.append((obs, action, chunk, seed_offset))
                
                # Submit all chunks to worker pool
                results = []
                for obs_chunk, action_chunk, chunk_size, seed_offset in chunk_tasks:
                    result = pool.apply_async(
                        worker_run_chunk, 
                        args=(self, obs_chunk, action_chunk, chunk_size, seed_offset)
                    )
                    results.append(result)
                
                # Collect results and check for early stopping
                for i, result in enumerate(results):
                    try:
                        chunk_result = result.get(timeout=30)  # 30 second timeout
                        if len(chunk_result) == 3:
                            chunk_total, chunk_runs, chunk_individual_results = chunk_result
                        else:
                            # Backward compatibility for old format
                            chunk_total, chunk_runs = chunk_result
                            chunk_individual_results = None
                            
                        total += chunk_total
                        runs_done += chunk_runs
                        
                        # Store individual results for variance reduction if available
                        if simulation_results is not None and chunk_individual_results is not None:
                            simulation_results.extend(chunk_individual_results)
                        
                        # Check early stopping after each chunk
                        if best_mean is not None and runs_done > 8:
                            mean = total / runs_done
                            if mean + early_margin < best_mean:
                                # Cancel remaining results
                                for j in range(i+1, len(results)):
                                    try:
                                        results[j].get(timeout=0.1)
                                    except:
                                        pass
                                break
                    except Exception:
                        # If a worker fails, fall back to sequential for this chunk
                        fallback_result = self._run_chunk_simulations(obs, action, chunk, i * chunk * 1000)
                        if len(fallback_result) == 3:
                            chunk_total, chunk_runs, chunk_individual_results = fallback_result
                        else:
                            chunk_total, chunk_runs = fallback_result
                            chunk_individual_results = None
                            
                        total += chunk_total
                        runs_done += chunk_runs
                        
                        # Store individual results for variance reduction if available
                        if simulation_results is not None and chunk_individual_results is not None:
                            simulation_results.extend(chunk_individual_results)

            # Handle remainder sequentially (usually small)
            if rem > 0:
                remainder_result = self._run_chunk_simulations(obs, action, rem, full_chunks * chunk * 1000)
                if len(remainder_result) == 3:
                    remainder_total, remainder_runs, remainder_individual_results = remainder_result
                else:
                    remainder_total, remainder_runs = remainder_result
                    remainder_individual_results = None
                    
                total += remainder_total
                runs_done += remainder_runs
                
                # Store individual results for variance reduction if available
                if simulation_results is not None and remainder_individual_results is not None:
                    simulation_results.extend(remainder_individual_results)

        except Exception:
            # On any error, fall back to sequential evaluation
            return self._evaluate_action_sequential(obs, action, best_so_far)

        if runs_done == 0:
            return 0.0
            
        # Apply variance reduction if enabled
        if self.variance_reduction and simulation_results:
            from .mc_utils import apply_variance_reduction
            return apply_variance_reduction(simulation_results, obs, action)
        else:
            return total / runs_done

    def _initialize_parallel_parameters(self, enable_parallel, num_workers):
        """Initialize parallel processing parameters during agent construction."""
        # Parallel processing parameters
        self.enable_parallel = bool(enable_parallel)
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(1, int(self.num_workers))
        elif self.enable_parallel:
            # Auto-detect number of workers
            self.num_workers = min(max(1, mp.cpu_count() - 1), 4)  # Leave 1 core free, max 4 workers
        
        # Worker pool for parallel processing (initialized when needed)
        self._worker_pool = None