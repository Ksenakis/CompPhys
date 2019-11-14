#include <stdio.h>
#include <cmath>
#include <fstream>
#include <string.h>
using namespace std;



int run(int argc, char* argv[]){
printf("I am run \n");

int verbose = 0;
string out_file_name = "out.txt";

for(int i_arg=1;i_arg<argc;i_arg++){
    if(strcmp(argv[i_arg], "-v")==0){verbose=atoi(argv[i_arg+1]);i_arg++;printf("Verbose set to %i\n", verbose);}
    if(strcmp(argv[i_arg], "-o")==0){out_file_name=string(argv[i_arg+1]);i_arg++;}
}
return 0;
}

int main(int argc, char* argv[]){
    printf("I am main \n");
    clock_t beginglobal = clock();
    ofstream runfile;
    runfile.open("output.txt");
    runfile << "In main\n";
    runfile.close();

    run(argc, argv);

    clock_t endglobal = clock();
    double elapsed_secs = double(endglobal - beginglobal) / CLOCKS_PER_SEC;
    
    printf("Time of computing: %f seconds = %f minutes = too long\n", elapsed_secs, elapsed_secs/60);
    return 0;
}
