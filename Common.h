#pragma once

typedef double R; // real number datatype;

#define SAFE_DELETE(pointer) if(pointer != nullptr){delete pointer; pointer=nullptr;}
#define SAFE_DELETE_ARRAY(pointer) if(pointer != nullptr){delete [] pointer; pointer=nullptr;}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))