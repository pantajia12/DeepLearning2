#include <stdio.h>
#include <malloc.h>
#include <math.h>

double** forward(double*);
double** dot(double*, double*, int, int, int);
double** add(double**, double*, int, int);
double** sigmoid(double**, int, int);
double** identity_function(double**);

double W1[2][3] = {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}};
double b1[1][3] = {0.1, 0.2, 0.3};
double W2[3][2] = {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}};
double b2[1][2] = {0.1, 0.2};
double W3[2][2] = {{0.1, 0.3}, {0.2, 0.4}};
double b3[1][2] = {0.1, 0.2};

int main(){
    double x[1][2] = {1.0, 0.5};
    double** y = forward(&x);
    printf("%lf %lf",y[0][0], y[0][1]);
}

double** forward(double* x){
    double** a1 = add((dot(x, W1, 1, 2, 3)), b1, 1, 3);
    double** z1 = sigmoid(a1, 1, 3);
    double** a2 = add((dot(*z1, W2, 1, 3, 2)), b2, 1, 2);
    double** z2 = sigmoid(a2, 1, 2);
    double** a3 = add((dot(*z2, W3, 1, 2, 2)), b3, 1, 2);
    double** y = identity_function(a3);
    
    return y;
} 

double** dot(double* x, double* W, int l, int n, int m){//(x, W1, x[n][], x[][n], W1[][n])
    double **result; // l행 m열로 선언할거임

    result = calloc(sizeof(double*), l);
    result[0] = calloc(sizeof(double), l * m);
    for (int i = 1; i < l; ++i) result[i] = result[i - 1] + m;

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){ 
            for(int k = 0; k < n; k++){
                result[i][j] += x[k+i*n] * W[j+k*m];//2차원 배열이 넘어오면서 1차원으로 변함
            }
        }
    }

    return result;
}

double** add(double** a, double* b, int x, int y){
	double **result; // x행 y열로 선언할거임

    result = calloc(sizeof(double*), x);
    result[0] = calloc(sizeof(double), x * y);
    for (int i = 1; i < x; ++i) result[i] = result[i - 1] + y;

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] + b[m+n*y];
        }
    }

    return result;
}

double** sigmoid(double** x, int l, int m){
    double **result; // l행 m열로 선언할거임

    result = calloc(sizeof(double*), l);
    result[0] = calloc(sizeof(double), l * m);
    for (int i = 1; i < l; ++i) result[i] = result[i - 1] + m;

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = 1 / (1 + pow(exp(1.0), (-x[i][j])));
        }
    }

    return result;
}

double** identity_function(double** x){
    return x;
}