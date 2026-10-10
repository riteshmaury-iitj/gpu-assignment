/* ccl_cpu.c -- sequential reference for Connected-Component Labeling (8-connectivity).
 * Two modes:
 *   --algo bfs   : flood-fill with an explicit queue (fast, used for verification and CPU baseline)
 *   --algo prop  : the SAME iterative min-label propagation the GPU runs, executed serially
 *                  (reports iteration count; shows why the GPU version needs parallelism)
 * Build: gcc -O3 -o ccl_cpu ccl_cpu.c            (MSVC: cl /O2 ccl_cpu.c)
 * Usage: ./ccl_cpu --image img.bin [--algo bfs|prop] [--out-labels l.bin] [--out-stats s.csv]
 * Same file formats as ccl_cuda.cu; labels are 1..K in first-encounter (raster) order.       */
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static double now_ms(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1e3*(double)c.QuadPart/(double)f.QuadPart; }
#else
  static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
#endif

int main(int argc, char** argv){
    const char *img=NULL,*ol="labels_cpu.bin",*os="stats_cpu.csv",*algo="bfs";
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--image")) img=argv[++a];
        else if(!strcmp(argv[a],"--algo")) algo=argv[++a];
        else if(!strcmp(argv[a],"--out-labels")) ol=argv[++a];
        else if(!strcmp(argv[a],"--out-stats")) os=argv[++a];
    }
    if(!img){fprintf(stderr,"usage: --image img.bin [--algo bfs|prop]\n");return 1;}
    FILE* f=fopen(img,"rb"); if(!f){perror("image");return 1;}
    int W,H; if(fread(&W,4,1,f)!=1||fread(&H,4,1,f)!=1) return 1;
    long N=(long)W*H; unsigned char* I=malloc(N); if(fread(I,1,N,f)!=(size_t)N) return 1; fclose(f);
    int* L=calloc(N,sizeof(int)); int K=0, iters=0;
    double t0=now_ms();

    if(!strcmp(algo,"bfs")){
        long* q=malloc(N*sizeof(long));
        for(long s=0;s<N;s++){
            if(!I[s]||L[s]) continue;
            K++; long qh=0,qt=0; q[qt++]=s; L[s]=K;
            while(qh<qt){
                long p=q[qh++]; int x=(int)(p%W), y=(int)(p/W);
                for(int dy=-1;dy<=1;dy++){ int ny=y+dy; if(ny<0||ny>=H) continue;
                    for(int dx=-1;dx<=1;dx++){ int nx=x+dx; if(nx<0||nx>=W||(!dx&&!dy)) continue;
                        long n=(long)ny*W+nx; if(I[n]&&!L[n]){L[n]=K;q[qt++]=n;} } }
            }
        }
        free(q);
    } else { /* serial version of the GPU min-label propagation, for iteration counts */
        for(long i=0;i<N;i++) L[i]=I[i]?(int)i:0;
        int changed=1;
        while(changed){ changed=0; iters++;
            for(long p=0;p<N;p++){ if(!I[p]) continue; int x=(int)(p%W), y=(int)(p/W); int best=L[p];
                for(int dy=-1;dy<=1;dy++){ int ny=y+dy; if(ny<0||ny>=H) continue;
                    for(int dx=-1;dx<=1;dx++){ int nx=x+dx; if(nx<0||nx>=W||(!dx&&!dy)) continue;
                        long n=(long)ny*W+nx; if(I[n]&&L[n]<best) best=L[n]; } }
                if(best<L[p]){L[p]=best;changed=1;} }
            if(iters%4==0) for(long p=0;p<N;p++){ if(!I[p]) continue; int l=L[p]; while(L[l]!=l) l=L[l]; L[p]=l; }
        }
        for(long p=0;p<N;p++){ if(!I[p]) continue; int l=L[p]; while(L[l]!=l) l=L[l]; L[p]=l; }
        int* id=calloc(N,sizeof(int));
        for(long p=0;p<N;p++) if(I[p]){ int r=L[p]; if(!id[r]) id[r]=++K; }
        for(long p=0;p<N;p++) L[p]=I[p]?id[L[p]]:0;
        free(id);
    }
    double ms=now_ms()-t0;

    /* statistics */
    long *area=calloc(K+1,8),*sx=calloc(K+1,8),*sy=calloc(K+1,8);
    int *x0=malloc((K+1)*4),*y0=malloc((K+1)*4),*x1=malloc((K+1)*4),*y1=malloc((K+1)*4);
    for(int k=0;k<=K;k++){x0[k]=W;y0[k]=H;x1[k]=-1;y1[k]=-1;}
    for(long p=0;p<N;p++){ int k=L[p]; if(!k) continue; int x=(int)(p%W),y=(int)(p/W);
        area[k]++; sx[k]+=x; sy[k]+=y; if(x<x0[k])x0[k]=x; if(y<y0[k])y0[k]=y; if(x>x1[k])x1[k]=x; if(y>y1[k])y1[k]=y; }
    double ms_total=now_ms()-t0;

    f=fopen(ol,"wb"); fwrite(&W,4,1,f); fwrite(&H,4,1,f); fwrite(L,4,N,f); fclose(f);
    f=fopen(os,"w"); fprintf(f,"id,area,cx,cy,xmin,ymin,xmax,ymax\n");
    for(int k=1;k<=K;k++) fprintf(f,"%d,%ld,%.4f,%.4f,%d,%d,%d,%d\n",k,area[k],(double)sx[k]/area[k],(double)sy[k]/area[k],x0[k],y0[k],x1[k],y1[k]);
    fclose(f);
    long fg=0; for(long i=0;i<N;i++) fg+=I[i]?1:0;
    printf("RESULT mode=cpu-%s W=%d H=%d N=%ld fg=%ld components=%d iters=%d time_ms=%.3f label_ms=%.3f stats_ms=%.3f\n",
           algo,W,H,N,fg,K,iters,ms_total,ms,ms_total-ms);
    return 0;
}
