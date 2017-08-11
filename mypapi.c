// create by Zhang Pengfei 
//
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "papi.h"
#include "util.h"

#define MAX_RAPL_EVENETS 64

static int EventSet = PAPI_NULL;
//static long long * values;
static char event_names[MAX_RAPL_EVENETS][PAPI_MAX_STR_LEN];
static char units[MAX_RAPL_EVENETS][PAPI_MIN_STR_LEN];
static int data_type[MAX_RAPL_EVENETS];
static int num_events;
static PAPI_event_info_t evinfo;

/* RAPL Library init */
int RAPL_init(long long ** values){
	
	int retval, cid, rapl_cid = -1, numcmp;
	int code;
	int r;
	const PAPI_component_info_t * cmpinfo = NULL;

	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT ){
		fprintf(stderr, "PAPI library init failed!\n");
		exit(1);
	}
	numcmp = PAPI_num_components();

	for(cid = 0; cid < numcmp; cid++){
		
		if((cmpinfo = PAPI_get_component_info(cid)) == NULL){
			fprintf(stderr, "PAPI_get_component_info failed!\n");
			exit(1);
		}
		if(strstr(cmpinfo->name, "rapl")){
			rapl_cid = cid;
			printf("Found rapl component at cid %d\n", rapl_cid);
			if( cmpinfo->disabled ){
				fprintf(stderr, "rapl disabled!\n");
				exit(1);
			}
			break;
		}
	}
	if( cid == numcmp ){
		fprintf(stderr, "rapl not found!\n");
		exit(1);
	}

	retval = PAPI_create_eventset( &EventSet );
	if(retval != PAPI_OK ){
		fprintf(stderr, "PAPI_create_eventset failed!\n");
		exit(1);
	}
	
	code = PAPI_NATIVE_MASK;
	r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, rapl_cid );

	while( r==PAPI_OK ){
		retval = PAPI_event_code_to_name( code, event_names[num_events] );
		if( retval != PAPI_OK ){
			fprintf(stderr,"PAPI_event_code_to_name failed!\n");
			exit(1);
		}

		retval = PAPI_get_event_info(code, &evinfo);
		if( retval != PAPI_OK ){
			fprintf(stderr,"PAPI_get_event_info failed!\n");
			exit(1);
		}

		strncpy(units[num_events],evinfo.units,sizeof(units[0])-1);
		units[num_events][sizeof(units[0])-1] = '\0';
		data_type[num_events] = evinfo.data_type;
		retval = PAPI_add_event( EventSet, code );
		
		if( retval != PAPI_OK )
			break;
		num_events ++;
		r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, rapl_cid );
	}

	(*values) = calloc(num_events, sizeof(long long));
	printf("number of events: %d\n", num_events);
	if( *values == NULL ) {
		fprintf(stderr, "values calloc failed!\n");
		exit(1);
	}

	return EventSet;
}

struct Energy 
energy_cost(int print, long long * values){
	int i;
	double total_energy = 0;
	double total_dram = 0;
	double total_package = 0;
	for( i = 0; i < num_events; i++ ){
		if(strstr(units[i], "nJ")) {	
			total_energy += (double)values[i] / 1.0e9;
			if(strstr(event_names[i], "DRAM_ENERGY")){
				total_dram += (double)values[i] / 1.0e9;
			}
			if(strstr(event_names[i], "PACKAGE_ENERGY")) {
				total_package += (double)values[i] / 1.0e9;
			}

			if(print) {
				printf("%-40s%12.6fJ \n",
						event_names[i],
						(double)values[i]/1.0e9);
			}
		}
	}
	if(print)
		fprintf(stderr, "total energy: %fJ, total package: %fJ, total DRAM: %fJ\n", 
				total_energy, total_package, total_dram);
	
	struct Energy energy;
	energy.total_energy = total_energy;
	energy.total_dram = total_dram;
	energy.total_package = total_package;
	return energy;
}



	
		

