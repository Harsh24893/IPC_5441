#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<time.h>
#include<pthread.h>

#define NS_PER_US 1000

typedef struct{
	int id;
	int x;
	int y;
	int height;
	int width;
	int top_n;
	int * top_n_ids;
	int bottom_n;
	int *bottom_n_ids;
	int left_n;
	int *left_n_ids;
	int right_n;
	int *right_n_ids;
	
	double dsv;
}box;

box* boxes;
double* weighted_avg_adjacent_temp;

double AFFECT_RATE = 0;
double EPSILON = 0;

int NUMBER_OF_THREADS = 0;
int num_grid_boxes = 0;
int *cyclic_index;
int numberOfIterations;

double maxDSV;
double minDSV;

bool convergenceAchieved = false;

pthread_barrier_t   barrier;

void readFile() 
{
	int num_rows = 0;
	int num_columns = 0;
	
	scanf("%d%d%d", &num_grid_boxes, &num_rows, &num_columns);
	
	boxes = (box *)calloc(num_grid_boxes, sizeof(box));
	weighted_avg_adjacent_temp = (double*)calloc(num_grid_boxes, sizeof(double));
	
	int i;
	for(i = 0; i < num_grid_boxes; i++) {
		scanf("%d", &(boxes[i].id));
       	scanf("%d%d%d%d", &(boxes[i].y), &(boxes[i].x), &(boxes[i].height), &(boxes[i].width));

		scanf("%d", &(boxes[i].top_n));
		boxes[i].top_n_ids = (int *)calloc(boxes[i].top_n, sizeof(int));
		int j;
		for(j = 0; j < boxes[i].top_n; j++){
			scanf("%d", &(boxes[i].top_n_ids[j]));
		}
		
		scanf("%d", &(boxes[i].bottom_n));
		boxes[i].bottom_n_ids = (int *)calloc(boxes[i].bottom_n, sizeof(int));
		for(j = 0; j < boxes[i].bottom_n; j++){
			scanf("%d", &(boxes[i].bottom_n_ids[j]));
		}
		
		scanf("%d", &(boxes[i].left_n));
		boxes[i].left_n_ids = (int *)calloc(boxes[i].left_n, sizeof(int));
		for(j = 0; j < boxes[i].left_n; j++){
			scanf("%d", &(boxes[i].left_n_ids[j]));
		}
		
		scanf("%d", &(boxes[i].right_n));
		boxes[i].right_n_ids = (int *)calloc(boxes[i].right_n, sizeof(int));
		for(j = 0; j < boxes[i].right_n; j++){
			scanf("%d", &(boxes[i].right_n_ids[j]));
		}
		
		scanf("%lf", &(boxes[i].dsv));
	}
	
	cyclic_index = malloc(sizeof(int) * NUMBER_OF_THREADS);
    int index;
	for(index = 0; index < NUMBER_OF_THREADS; index++) {
		cyclic_index[index] = index;
	}
}

bool isConvergenceAchieved()
{
	maxDSV = 0;
	minDSV = boxes[1].dsv;
	int i;
	for (i = 0; i < num_grid_boxes; i++)// find the min max DSV
	{
		if (boxes[i].dsv > maxDSV)
			maxDSV = boxes[i].dsv;
		if (boxes[i].dsv < minDSV)
			minDSV = boxes[i].dsv;
	}
	double tempDifference = maxDSV - minDSV;// calculate temp difference
	if (tempDifference <= maxDSV*EPSILON)// check for convergence
		return true;
	else
		return false;
}

int calculateCommonPerimeter(int mySide1, int mySide2, int NeighborSide1, int NeighborSide2)
{
	int maxSide1 = 0;
	int minSide2 = 0;

	if (mySide1 > NeighborSide1)
		maxSide1 = mySide1;
	else
		maxSide1 = NeighborSide1;

	if (mySide2 < NeighborSide2)
		minSide2 = mySide2;
	else
		minSide2 = NeighborSide2;

	return(minSide2 - maxSide1);
}

double calculateAvgWeightedAdjecentTemp(int dsvID_in)
{
	double weightedTotal = 0;
	int commonPerimeter = 0;
	int i;
	if (boxes[dsvID_in].left_n)
	{
		for (i = 0; i < boxes[dsvID_in].left_n; i++)
		{
			int leftNeighborID = boxes[dsvID_in].left_n_ids[i];
			int NeighborRightUpperY = boxes[leftNeighborID].y;
			int NeighborRightLowerY = NeighborRightUpperY + boxes[leftNeighborID].height;
			commonPerimeter = calculateCommonPerimeter(boxes[dsvID_in].y, boxes[dsvID_in].y + boxes[dsvID_in].height, NeighborRightUpperY, NeighborRightLowerY);
			weightedTotal += commonPerimeter* boxes[leftNeighborID].dsv;
		}
	}
	else
		weightedTotal += boxes[dsvID_in].height*boxes[dsvID_in].dsv;// if no neighbour take outside temperature as that of the box

	if (boxes[dsvID_in].right_n)
	{
		for (i = 0; i < boxes[dsvID_in].right_n; i++)
		{
			int rightNeighborID = boxes[dsvID_in].right_n_ids[i];
			int NeighborLeftUpperY = boxes[rightNeighborID].y;
			int NeighborLeftLowerY = NeighborLeftUpperY + boxes[rightNeighborID].height;
			commonPerimeter = calculateCommonPerimeter(boxes[dsvID_in].y, boxes[dsvID_in].y + boxes[dsvID_in].height, NeighborLeftUpperY, NeighborLeftLowerY);
			weightedTotal += commonPerimeter* boxes[rightNeighborID].dsv;
		}
	}
	else
		weightedTotal += boxes[dsvID_in].height*boxes[dsvID_in].dsv;// if no neighbour take outside temperature as that of the box
	
	if (boxes[dsvID_in].top_n)
	{
		for (i = 0; i < boxes[dsvID_in].top_n; i++)
		{
			int topNeighborID = boxes[dsvID_in].top_n_ids[i];
			int NeighborLowerLeftX = boxes[topNeighborID].x;
			int NeighborLowerRightX = NeighborLowerLeftX + boxes[topNeighborID].width;
			commonPerimeter = calculateCommonPerimeter(boxes[dsvID_in].x, boxes[dsvID_in].x + boxes[dsvID_in].width, NeighborLowerLeftX, NeighborLowerRightX);
			weightedTotal += commonPerimeter* boxes[topNeighborID].dsv;
		}
	}
	else
		weightedTotal += boxes[dsvID_in].width*boxes[dsvID_in].dsv;// if no neighbour take outside temperature as that of the box
	
	if (boxes[dsvID_in].bottom_n)
	{
		for (i = 0; i < boxes[dsvID_in].bottom_n; i++)
		{
			int bottomNeighborID = boxes[dsvID_in].bottom_n_ids[i];
			int NeighborUpperLeftX = boxes[bottomNeighborID].x;
			int NeighborUpperRightX = NeighborUpperLeftX + boxes[bottomNeighborID].width;
			commonPerimeter = calculateCommonPerimeter(boxes[dsvID_in].x, boxes[dsvID_in].x + boxes[dsvID_in].width, NeighborUpperLeftX, NeighborUpperRightX);
			weightedTotal += commonPerimeter* boxes[bottomNeighborID].dsv;
		}
	}
	else
		weightedTotal += boxes[dsvID_in].width*boxes[dsvID_in].dsv;// if no neighbour take outside temperature as that of the box
	
	double weightedAvgTotal = weightedTotal / (2 * (boxes[dsvID_in].height + boxes[dsvID_in].width));
	return weightedAvgTotal;
}

void commitNewWeightedAvg()
{
	int i;
	for (i = 0; i < num_grid_boxes; i++)
	{
		if (boxes[i].dsv < weighted_avg_adjacent_temp[i])
			boxes[i].dsv = boxes[i].dsv + ((weighted_avg_adjacent_temp[i] - boxes[i].dsv)*AFFECT_RATE);
		else
			boxes[i].dsv = boxes[i].dsv - ((boxes[i].dsv - weighted_avg_adjacent_temp[i])*AFFECT_RATE);
	}
}

void * computeDSV(void* ptr)
{
	int t_id = *((int *) ptr);
	
	int avg_unit_box_per_thread = num_grid_boxes / NUMBER_OF_THREADS;
	int extra_boxes = num_grid_boxes % NUMBER_OF_THREADS;
	
	int number_boxes_current_thread = t_id < extra_boxes ? avg_unit_box_per_thread + 1 : avg_unit_box_per_thread;
	int start = t_id * avg_unit_box_per_thread + ((t_id < extra_boxes) ? t_id : extra_boxes);
	
	while(!convergenceAchieved)
	{
		  int index;
	      for (index = start; index < start+number_boxes_current_thread; index++)// calculate box_dsv for boxes assigned to each thread in parallel
	      {
		      weighted_avg_adjacent_temp[index] = calculateAvgWeightedAdjecentTemp(index);
		  }
	      pthread_barrier_wait (&barrier); 
	      
	      if(t_id == 0)
	      {
			commitNewWeightedAvg();
			++numberOfIterations;
			convergenceAchieved = isConvergenceAchieved();
	      }
	      pthread_barrier_wait (&barrier);
	}
	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
	if(argc < 4) {
        	fprintf(stderr, "Error in running the program\n");
        	return 0;
    } else if (argc > 4) {
        	fprintf(stderr, "Error in running the program\n");
        	return 0;
    }
	
    AFFECT_RATE = atof(argv[1]);
	EPSILON = atof(argv[2]);
	NUMBER_OF_THREADS = atof(argv[3]);
	
    readFile();
	numberOfIterations = 0;
	pthread_t threads[NUMBER_OF_THREADS];
	pthread_barrier_init (&barrier, NULL, NUMBER_OF_THREADS);
	
	//===============================Time Calculations=======================================
	clock_t clock_time = 0;
	time_t time_time=0;
	clock_t begin, end;
    time_t start, finish;
    struct timespec start_c, end_c; 
	double diff; 
	
	clock_gettime(CLOCK_REALTIME,& start_c);
    time(&start);
    begin = clock();
	//========================================================================================

	
		int index;
		for(index = 0; index < NUMBER_OF_THREADS; index++) {
			int rc = pthread_create(&threads[index], NULL, computeDSV, cyclic_index + index);
			if(rc)
			{
				printf("Error - pthread_create() return code: %d\n",rc);
				exit(EXIT_FAILURE);
			}
		}
		
		for(index = 0; index < NUMBER_OF_THREADS; index++) {
			pthread_join( threads[index], NULL);
		}
	

	//===============================Time Calculations=======================================
	end = clock();
    time(&finish);
    clock_gettime(CLOCK_REALTIME,& end_c); 
	
	diff = (double)( ((end_c.tv_sec - start_c.tv_sec)*CLOCKS_PER_SEC) + ((end_c.tv_nsec -start_c.tv_nsec)/NS_PER_US) );
    clock_time = end - begin;
    time_time = finish - start;
	//========================================================================================
	
	printf("%s","***********************************************************************\n");
	printf("Dissipation converged in %d iterations\n", numberOfIterations);
	printf("with max DSV = %lf and min DSV = %lf\n", maxDSV, minDSV);
	printf("Number of threads: %d, Affect rate = %lf , epsilon = %lf\n", NUMBER_OF_THREADS, AFFECT_RATE, EPSILON);
	printf("elapsed convergence loop time(clock) : %ld \n",clock_time);
	printf("elapsed convergence loop time(time) : %ld \n",time_time);
	printf("elapsed convergence loop time (chrono): %lf \n",diff);
	printf("***********************************************************************\n");
	
    return 0;
}