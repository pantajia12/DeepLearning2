#include <stdio.h>
#include <malloc.h>
#include <math.h>

double** reset();
double** forward(double**);
double** dot(double**, double*, int);
double** add(double**, double*, int);
double** sigmoid(double**);
double** softmax(double**);
double** relu(double**);
double** identity_function(double**);
double** createArray(int, int);
double arrMax(double**);

double W1[2][3] = {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}};
double b1[1][3] = {0.1, 0.2, 0.3};
double W2[3][2] = {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}};
double b2[1][2] = {0.1, 0.2};
double W3[2][2] = {{0.1, 0.3}, {0.2, 0.4}};
double b3[1][2] = {0.1, 0.2};

int main(){
    double** x = reset();
    double** y = forward(x);
    printf("%lf %lf",y[0][0], y[0][1]);
}

double** reset(){
    double** x = createArray(1, 2);
    x[0][0] = 1.0;
    x[0][1] = 0.5;
    return x;
}

double** forward(double** x){

    int l = _msize(x)/sizeof(x[0]); //x[n][]

    double** a1 = createArray(l, sizeof(*W1)/sizeof(double));
    a1 = add((dot(x, W1, sizeof(*W1)/sizeof(double))), b1, sizeof(*W1)/sizeof(double));

    double** z1 = createArray(_msize(a1)/sizeof(a1[0]), _msize(a1[0])/sizeof(a1[0][0])/(_msize(a1)/sizeof(a1[0])));
    z1 = sigmoid(a1);
    
    double** a2 = createArray(_msize(z1)/sizeof(z1[0]), sizeof(*W2)/sizeof(double));
    a2 = add((dot(z1, W2, sizeof(*W2)/sizeof(double))), b2, sizeof(*W2)/sizeof(double));
    
    double** z2 = createArray(_msize(a2)/sizeof(a2[0]), _msize(a2[0])/sizeof(a2[0][0])/(_msize(a2)/sizeof(a2[0])));
    z2 = sigmoid(a2);

    double** a3 = createArray(_msize(z2)/sizeof(z2[0]), sizeof(*W3)/sizeof(double));
    a3 = add((dot(z2, W3, sizeof(*W3)/sizeof(double))), b3, sizeof(*W3)/sizeof(double));

    double** y = createArray(_msize(a3)/sizeof(a3[0]), _msize(a3[0])/sizeof(a3[0][0])/(_msize(a3)/sizeof(a3[0])));
    y = softmax(a3);
    
    return y;
} 

double** dot(double** x, double* W, int m){//(x, W1, W1[][n])
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int n = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){ 
            for(int k = 0; k < n; k++){
                result[i][j] += x[i][k] * W[j+k*m];//2차원 배열이 넘어오면서 1차원으로 변함
            }
        }
    }

    return result;
}

double** add(double** a, double* b, int y){
    int x = _msize(a)/sizeof(a[0]); //a[n][]
    double **result = createArray(x, y);

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] + b[m+n*y];
        }
    }

    return result;
}

double** sigmoid(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = 1 / (1 + pow(exp(1.0), (-x[i][j])));
        }
    }

    return result;
}

double** softmax(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    double Max = arrMax(x);
    double sumExpX = 0;

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            sumExpX += exp(x[i][j] - Max);
        }
    }

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = exp(x[i][j] - Max) / sumExpX;
        }
    }

    return result;
}

double** relu(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = x[i][j] > 0 ? x[i][j] : 0;
        }
    }

    return result;
}

double** identity_function(double** x){
    return x;
}

double** createArray(int l, int m){
    double **result; // l행 m열로 선언할거임

    result = calloc(sizeof(double*), l);
    result[0] = calloc(sizeof(double), l * m);
    for (int i = 1; i < l; ++i) result[i] = result[i - 1] + m;

    return result;
}

double arrMax(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double max = x[0][0];

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            if(max < x[i][j]){
                max = x[i][j];
            }
        }
    }

    return max;
}