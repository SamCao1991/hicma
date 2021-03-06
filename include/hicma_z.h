/**
 * @copyright (c) 2017 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 **/
/**
 * @file hicma_z.h
 *
 *  HiCMA computational routines
 *  HiCMA is a software package provided by King Abdullah University of Science and Technology (KAUST)
 *
 * @version 0.1.1
 * @author Kadir Akbudak
 * @date 2018-11-08
 **/
#ifndef _HICMA_Z_H_
#define _HICMA_Z_H_

#undef REAL
#define COMPLEX
#include "morse.h"
#include "hicma.h"

#ifdef __cplusplus
extern "C" {
#endif


//FIXME Naive interfaces taking only arrays are not implemented yet
int HICMA_zpotrf(MORSE_enum uplo, int N, double *A, int LDA);
int HICMA_zpotrf_Tile(MORSE_enum uplo,
        MORSE_desc_t *AUV, MORSE_desc_t *AD, MORSE_desc_t *Ark,
        int rk, int maxrk, double acc
        );
int HICMA_zpotrf_Tile_Async(MORSE_enum uplo,
        MORSE_desc_t *AUV, MORSE_desc_t *AD, MORSE_desc_t *Ark,
        int rk, int maxrk, double acc,
        MORSE_sequence_t *sequence, MORSE_request_t *request );
int HICMA_zgemm_Tile(MORSE_enum transA, MORSE_enum transB,
        double alpha,
        MORSE_desc_t *AUV, MORSE_desc_t *Ark,
        MORSE_desc_t *BUV, MORSE_desc_t *Brk,
        double beta,
        MORSE_desc_t *CUV, MORSE_desc_t *Crk ,
        int rk,
        int maxrk,
        double acc
        );
int HICMA_zgemm_Tile_Async(MORSE_enum transA, MORSE_enum transB,
        double alpha,
        MORSE_desc_t *AUV, MORSE_desc_t *Ark,
        MORSE_desc_t *BUV, MORSE_desc_t *Brk,
        double beta,
        MORSE_desc_t *CUV, MORSE_desc_t *Crk,
        int rk,
        int maxrk,
        double acc ,
        MORSE_sequence_t *sequence, MORSE_request_t *request);
int HICMA_zgytlr(
        MORSE_enum   uplo,
        int M, int N,
        double *AUV,
        double *AD,
        double *Ark,
        int LDA, unsigned long long int seed,
        int maxrank, double tol
        );
int HICMA_zgytlr_Tile(
        MORSE_enum   uplo,
        MORSE_desc_t *AUV,
        MORSE_desc_t *AD,
        MORSE_desc_t *Ark,
        unsigned long long int seed,
        int maxrank,
        double tol,
        int compress_diag,
        MORSE_desc_t *Dense
        );
int HICMA_zgytlr_Tile_Async(
        MORSE_enum   uplo,
        MORSE_desc_t *AUV,
        MORSE_desc_t *AD,
        MORSE_desc_t *Ark,
        unsigned long long int seed,
        int maxrank, double tol,
        int compress_diag,
        MORSE_desc_t *Dense,
        MORSE_sequence_t *sequence, MORSE_request_t *request );
int HICMA_zhagcm_Tile(
        MORSE_enum   uplo,
        MORSE_desc_t *AUV,
        MORSE_desc_t *Ark,
        int numrows_matrix,
        int numcols_matrix,
        int numrows_block,
        int numcols_block,
        int maxrank,
        double tol
        );
int HICMA_zhagcm_Tile_Async(
        MORSE_enum   uplo,
        MORSE_desc_t *AUV,
        MORSE_desc_t *Ark,
        int numrows_matrix,
        int numcols_matrix,
        int numrows_block,
        int numcols_block,
        int maxrank, double tol,
        MORSE_sequence_t *sequence, MORSE_request_t *request );
int HICMA_zhagdm_Tile(
        MORSE_enum   uplo,
        MORSE_desc_t *Dense
        );
int HICMA_zhagdm_Tile_Async(
        MORSE_enum       uplo,
        MORSE_desc_t *Dense,
        MORSE_sequence_t *sequence,
        MORSE_request_t  *request);
int HICMA_zhagdmdiag_Tile(
        MORSE_enum   uplo,
        MORSE_desc_t *Dense
        );
int HICMA_zhagdmdiag_Tile_Async(
        MORSE_enum       uplo,
        MORSE_desc_t *Dense,
        MORSE_sequence_t *sequence,
        MORSE_request_t  *request);
int HICMA_ztrsm_Tile(MORSE_enum side, MORSE_enum uplo,
        MORSE_enum transA, MORSE_enum diag,
        double alpha, 
        MORSE_desc_t *AUV, 
        MORSE_desc_t *AD, 
        MORSE_desc_t *Ark, 
        MORSE_desc_t *BUV,
        MORSE_desc_t *Brk,
        int rk,
        int maxrk,
        double acc
        );
int HICMA_ztrsm_Tile_Async(MORSE_enum side, MORSE_enum uplo,
        MORSE_enum transA, MORSE_enum diag,
        double alpha, 
        MORSE_desc_t *AUV, 
        MORSE_desc_t *AD, 
        MORSE_desc_t *Ark, 
        MORSE_desc_t *BUV,
        MORSE_desc_t *Brk,
        int rk,
        int maxrk,
        double acc,
        MORSE_sequence_t *sequence, MORSE_request_t *request);
int HICMA_ztrsmd_Tile(MORSE_enum side, MORSE_enum uplo,
        MORSE_enum transA, MORSE_enum diag,
        double alpha, 
        MORSE_desc_t *AUV, 
        MORSE_desc_t *AD, 
        MORSE_desc_t *Ark, 
        MORSE_desc_t *Bdense,
        int maxrk
        );
int HICMA_ztrsmd_Tile_Async(MORSE_enum side, MORSE_enum uplo,
        MORSE_enum transA, MORSE_enum diag,
        double alpha, 
        MORSE_desc_t *AUV, 
        MORSE_desc_t *AD, 
        MORSE_desc_t *Ark, 
        MORSE_desc_t *Bdense,
        int maxrk,
        MORSE_sequence_t *sequence, MORSE_request_t *request);
int HICMA_zuncompress(
        MORSE_enum uplo, MORSE_desc_t *AUV, MORSE_desc_t *AD, MORSE_desc_t *Ark);
int HICMA_zuncompress_custom_size(MORSE_enum uplo,
        MORSE_desc_t *AUV, MORSE_desc_t *AD, MORSE_desc_t *Ark,
        int numrows_matrix,
        int numcolumns_matrix,
        int numrows_block,
        int numcolumns_block
        );
int HICMA_zdiag_vec2mat(
        MORSE_desc_t *vec, MORSE_desc_t *mat);
void HICMA_zgenerate_problem(
        int probtype, //problem type defined in hicma_constants.h 
        char sym,     // symmetricity of problem: 'N' or 'S'
        double decay, // decay of singular values. Will be used in HICMA_STARSH_PROB_RND. Set 0 for now.
        int _M,       // number of rows/columns of matrix
        int _nb,      // number of rows/columns of a single tile
        int _mt,      // number of tiles in row dimension
        int _nt,      // number of tiles in column dimension
        HICMA_problem_t *hicma_problem // pointer to hicma struct (starsh format will be used to pass coordinate info to number generation and compression phase)
        );
int HICMA_zgenrhs_Tile(
        MORSE_desc_t *A);
int HICMA_zgenrhs_Tile_Async(
        MORSE_desc_t     *A,
        MORSE_sequence_t *sequence,
        MORSE_request_t  *request);
void hicma_pzgenrhs(
        MORSE_desc_t *A,
        MORSE_sequence_t *sequence, MORSE_request_t *request );
#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
