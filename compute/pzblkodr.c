/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file pzblkodr.c
 *
 *  HiCMA auxiliary routines
 *  HiCMA is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.0.1
 * @author Jian Cao
 * @date 2021-03-01
 **/

#include "morse.h"
#include "hicma.h"
#include "hicma_common.h"
#include "control/common.h"
#include "hicma_runtime_z.h"
#include "coreblas/lapacke.h"

#include "control/hicma_config.h"
#include <stdio.h>
#include <algorithm>

extern int store_only_diagonal_tiles;
extern int print_index;
int pzblkodr_print_index = 0;
extern int print_mat;
extern int run_org;
int extra_barrier = 0;
/***************************************************************************//**
 *  Parallel tile iterative block reordering - dynamic scheduling
 **/
void hicma_pzblkodr(MORSE_enum uplo,
        MORSE_desc_t *AUV,
        MORSE_desc_t *AD,
        MORSE_desc_t *ADCp,
        MORSE_desc_t *Ark,
	MORSE_desc_t *a,
        MORSE_desc_t *b,
	MORSE_desc_t *aCp,
        MORSE_desc_t *bCp,
        MORSE_desc_t *y,
        MORSE_desc_t *p,
        MORSE_desc_t *idx,
        MORSE_sequence_t *sequence, MORSE_request_t *request,
        int rk, int maxrk, double acc)
{
    MORSE_context_t *morse;
    MORSE_option_t options;

    int k, m, n;
    size_t ws_host   = 0;
    size_t ws_worker   = 0;

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    morse = morse_context_self();
    if (sequence->status != MORSE_SUCCESS)
        return;
    RUNTIME_options_init(&options, morse, sequence, request);


/*#ifdef CHAMELEON_USE_MAGMA*/
    /*if (0) [> Disable the workspace as long as it is is not used (See StarPU codelet) <]*/
    /*{*/
        /*int nb = MORSE_IB; [> Approximate nb for simulation <]*/
/*#if !defined(CHAMELEON_SIMULATION)*/
        /*nb = magma_get_zpotrf_nb(AD->nb);*/
/*#endif*/
        /*ws_host = sizeof(double)*nb*nb;*/
    /*}*/
/*#endif*/
    //printf("%s %s %d maxrank=%d\n", __FILE__, __func__, __LINE__, maxrk);
    ws_worker =  //FIXME tentative size. FInd exact size. I think syrk uses less memory
        //Ali says: this workspace need to be fixed, not all tasks below need it nor need that much
        2 * AD->mb * 2 * maxrk   // for copying CU and CV into temporary buffer instead of using CUV itself. There is 2*maxrk because these buffers will be used to put two U's side by side
        + 2 * AD->mb       // qrtauA qrtauB
        + maxrk * maxrk    // qrb_aubut  AcolBcolT
        + 2 * AD->mb * maxrk // newU newV
        + (2*maxrk) * (2*maxrk)    // svd_rA     _rA
        //+ maxrk * maxrk    // svd_rB     _rB   I assume that use_trmm=1 so I commented out
        //+ maxrk * maxrk    // svd_T      _T    I assume that use_trmm=1 so I commented out
        + (2*maxrk)          // sigma
	+ 4 * AD->mb         // a, b, y, idx
        #ifdef HCORE_GEMM_USE_ORGQR
        + CUV->mb * 2*maxrk // newUV gemms
        #endif
        ;
    if(HICMA_get_use_fast_hcore_zgemm() == 1){
        double work_query;
        int lwork = -1;
        int info = LAPACKE_dgesvd_work( LAPACK_COL_MAJOR, 'A', 'A',
            2*maxrk, 2*maxrk,
            NULL, 2*maxrk,
            NULL,
            NULL, 2*maxrk,
            NULL, 2*maxrk, &work_query, lwork );
        lwork = (int)work_query;
        ws_worker += lwork; // superb
    }else{
        ws_worker += (2*maxrk); // superb
    }

    ws_worker *= sizeof(double); //FIXME use MORSE_Complex64_t
    //FIXME add ws_worker and ws_host calculation from compute/pzgeqrf.c when GPU/MAGMA is supported
    RUNTIME_options_ws_alloc( &options, ws_worker, ws_host );


    /*
     *  MorseLower
     */
    if (uplo == MorseLower) {
        for (k = 0; k < AD->mt; k++) {
            RUNTIME_iteration_push(morse, k);

            int tempkmd; 
            int ldakd;
	    int ldakd_min;
            int ADicol;
            int ADicol_min;

            //options.priority = 2*AD->mt - 2*k;
            options.priority = 5;

	    // uni odr for each diag blk
            for (m = k; m < AD->mt; m++) {
                tempkmd = m == AD->mt-1 ? AD->m-m*AD->mb : AD->mb;
                ldakd = BLKLDD(AD, m);
                if(store_only_diagonal_tiles == 1){
                    ADicol = 0;
                } else {
                    ADicol = m;
                }
                HICMA_TASK_zblkodr(
                        &options,
                        MorseLower, tempkmd, AD->mb,
                        AD, a, b, ADCp, aCp, bCp, idx, p, m, ADicol, ldakd, 0);
	    }
	    
	    // find which.min(p)
	    if(RUNTIME_desc_acquire(p) != MORSE_SUCCESS) {
		printf("Runtime acquire for p failed\n");
		return;
	    }
	    double *p_data = (double *)p->mat;
	    int min_index = std::min_element(p_data, p_data + AD->mt) - p_data;
	    RUNTIME_desc_release(p);

	    // block-wise switch
	    if(min_index != k){
                if(store_only_diagonal_tiles == 1){
		    ADicol = 0;
                    ADicol_min = 0;
                } else {
		    ADicol = k;
                    ADicol_min = min_index;
                }
                ldakd = BLKLDD(AD, k);
                ldakd_min = BLKLDD(AD, min_index);
	        HICMA_TASK_zswitch(&options, AD->mb, AD, k, min_index, ADicol, 
	          ADicol_min, ldakd, ldakd_min, 0);
		// continue here
	    }

            if(pzblkodr_print_index){
                printf("POTRF\t|tempkmd:%d k:%d ldakd:%d\n", tempkmd, k, ldakd);
            }
            for (m = k+1; m < AD->mt; m++) {
                int ldamuv = BLKLDD(AUV, m);

                //options.priority = 2*AD->mt - 2*k - m;
                options.priority = 4;
                if(pzblkodr_print_index){
                    printf("TRSM\t|m:%d k:%d ldakd:%d ldamuv:%d\n", m, k, ldakd, ldamuv);
                }
                /*
                 * X D^t = U V^t
                 * X = U V^t * inv(D^t)
                 * X = U (inv(D) * V )^t
                 * X = U * (trsm (lower, left, notranspose, D, V) )^t
                 * X = U * newV^t
                 */
                HICMA_TASK_ztrsm(
                        &options,
                        MorseLeft, MorseLower, MorseNoTrans, MorseNonUnit,
                        tempkmd, // number of rows of the diagonal block  
                        zone, AD, k, ADicol, 
                        ldakd,
                        AUV, m, k, 
                        ldamuv, 
                        Ark);
            }
            //MORSE_TASK_dataflush( &options, AV, k, k );
            //MORSE_TASK_dataflush( &options, AUV, k, k );
            RUNTIME_data_flush( sequence, AUV, k, k); 

            for (n = k+1; n < AD->mt; n++) {
                int tempnnd = n == AD->mt-1 ? AD->m-n*AD->mb : AD->mb;
                int ldand = BLKLDD(AD, n);
                int ldanuv = BLKLDD(AUV, n);
                int ADicol;
                if(store_only_diagonal_tiles == 1){
                    ADicol = 0;
                } else {
                    ADicol = n;
                }

                //options.priority = 2*AD->mt - 2*k - n;
                options.priority = 3;
                HICMA_TASK_zsyrk(
                        &options,
                        MorseLower, MorseNoTrans,
                        tempnnd,   0,
                        -1.0,
                        AUV, ldanuv,
                        Ark,
                        n, k,
                        1.0,
                        AD, ldand,
                        n, ADicol
                        );

                for (m = n+1; m < AD->mt; m++) {
                    int tempmmuv = m == AUV->mt-1 ? AUV->m - m*AUV->mb : AUV->mb;
                    int ldamuv = BLKLDD(AUV, m);

                    //options.priority = 2*AD->mt - 2*k - n - m;
                    options.priority = 2;
                    if(pzblkodr_print_index ){
                        printf("GEMM\t|A(%d,%d)=A(%d,%d)-A(%d,%d)*A(%d,%d) ldamuv:%d tempmmuv:%d\n", 
                                m,n,m,n,m,k,n,k,ldamuv, tempmmuv);
                    }
                    HICMA_TASK_zgemm(
                            &options,
                            MorseNoTrans, MorseTrans,
                            tempmmuv,
                            tempmmuv, 
                            mzone, AUV, Ark, m, k, ldamuv,
                            AUV, Ark, n, k, ldamuv,  
                            zone,  AUV, Ark, m, n, ldamuv,
                            rk, maxrk, acc);
                }
                //MORSE_TASK_dataflush( &options, AUV, n, k );
                RUNTIME_data_flush( sequence, AUV, n, k); 
            }
            RUNTIME_iteration_pop(morse);

            if(extra_barrier){
//                RUNTIME_barrier(morse);
            }
        }
    }
    /*
     *  MorseUpper
     */


    RUNTIME_options_ws_free(&options);
    RUNTIME_options_finalize(&options, morse);
}