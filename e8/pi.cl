__kernel void pi(const double step, const unsigned int steps_per_th, __local double* local_sums, __global double *reduction)
{
    const size_t id = get_global_id(0);
    const size_t l_id = get_local_id(0);
    const size_t l_size = get_local_size(0);
    const size_t grp   = get_group_id(0);

    unsigned int i = (id * steps_per_th);

    double x;
    double acc = 0.0;

    unsigned int num_steps = i + steps_per_th;

	for (i = i + 1; i <= num_steps; i++) {
		x = (i - 0.5) * step;
		acc += 4.0 / (1.0 + x * x);
	}

    local_sums[l_id] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (l_id != 0) return;

    for (i = 0, acc = 0.0; i < l_size; i++) 
        acc += local_sums[i];

    reduction[grp] = acc;
}

/*
static long num_steps = 100000000;
double step;
extern double wtime();   // returns time since some fixed past point (wtime.c)


int main ()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0/(double) num_steps;

    start_time =wtime();

    for (i=1;i<= num_steps; i++){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;
    run_time = wtime() - start_time;
    printf("\n pi with %ld steps is %lf in %lf seconds\n", num_steps, pi, run_time);
}
*/