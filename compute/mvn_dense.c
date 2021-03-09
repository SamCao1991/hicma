/**
 *
 * Copyright (c) 2017-2020  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ExaGeoStat is a software package provided by KAUST
 **/
/**
 *
 * @file MLE_exact.c
 *
 * ExaGeoStat dense computation main functions (i.e., compute MVN integration with SOV) 
 * @version 1.0.0
 *
 * @author Jian Cao
 * @date 2021-03-07
 *
 **/
#include "control/common.h"

//***************************************************************************************
int MORSE_mvn_Tile(MORSE_desc_t *Nrand, MORSE_desc_t *L, MORSE_desc_t *a, MORSE_desc_t *b,
    MORSE_desc_t *p, MORSE_desc_t *y, int &scalerIn)
{
    //Cholesky factorization for the Co-variance matrix
//    VERBOSE("Cholesky factorization of Sigma .....");
//    int success = MORSE_dpotrf_Tile(MorseLower, L);
//    SUCCESS(success, "Factorization cannot be performed..\n The matrix is not positive definite\n\n");
//    VERBOSE(" Done.\n");

    MORSE_context_t *morse;
    MORSE_sequence_t *sequence = NULL;
    MORSE_request_t request = MORSE_REQUEST_INITIALIZER;
    int status;

    morse = morse_context_self();
    if (morse == NULL) {
        morse_fatal_error("MORSE_mvn_Tile", "MORSE not initialized");
        return MORSE_ERR_NOT_INITIALIZED;
    }
    morse_sequence_create( morse, &sequence );

    MORSE_mvn_Tile_Async(Nrand, L, a, b, p, y, scalerIn, sequence, request);

    MORSE_Desc_Flush(Nrand, sequence);
    MORSE_Desc_Flush(L, sequence);
    MORSE_Desc_Flush(a, sequence);
    MORSE_Desc_Flush(b, sequence);
    MORSE_Desc_Flush(p, sequence);
    MORSE_Desc_Flush(y, sequence);

    morse_sequence_wait(morse, sequence);
    status = sequence->status;
    morse_sequence_destroy(morse, sequence);
    return status;
}


int MORSE_mvn_Tile_Async(MORSE_desc_t *Nrand, MORSE_desc_t *L, MORSE_desc_t *a, 
    MORSE_desc_t *b, MORSE_desc_t *p, MORSE_desc_t *y, int &scalerIn, 
    MORSE_sequence_t *sequence, MORSE_request_t *request)
{
    MORSE_context_t *morse;

    morse = morse_context_self();
    if (morse == NULL) {
        morse_fatal_error("MORSE_mvn_Tile", "MORSE not initialized");
        return MORSE_ERR_NOT_INITIALIZED;
    }
    if (sequence == NULL) {
        morse_fatal_error("MORSE_mvn_Tile", "NULL sequence");
        return MORSE_ERR_UNALLOCATED;
    }
    if (request == NULL) {
        morse_fatal_error("MORSE_mvn_Tile", "NULL request");
        return MORSE_ERR_UNALLOCATED;
    }
    /* Check sequence status */
    if (sequence->status == MORSE_SUCCESS) {
        request->status = MORSE_SUCCESS;
    }
    else {
        return morse_request_fail(sequence, request, MORSE_ERR_SEQUENCE_FLUSHED);
    }

    /* Check descriptors for correctness */
    if (morse_desc_check(Nrand) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid first descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    if (morse_desc_check(L) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid second descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    if (morse_desc_check(a) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid third descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    if (morse_desc_check(b) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid fourth descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    if (morse_desc_check(p) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid fifth descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    if (morse_desc_check(y) != MORSE_SUCCESS) {
        morse_error("MORSE_mvn_Tile", "invalid sixth descriptor");
        return morse_request_fail(sequence, request, MORSE_ERR_ILLEGAL_VALUE);
    } 
    /* Check input arguments */
    if(L->lm != L->ln){
    printf("Input L should be square\n");
    return(-1);
    }
    if(Nrand->lm != L->lm){
      printf("Row number of Nrand does not match L\n");
      return(-1);
    }
    if(a->lm != L->lm){
      printf("Row number of a does not match L\n");
      return(-1);
    }
    if(b->lm != L->lm){
      printf("Row number of b does not match L\n");
      return(-1);
    }
    if(p->lm != 1){
      printf("Row number of p should be one\n");
      return(-1);
    }
    if(y != NULL && y->lm != L->lm){
      printf("Row number of y does not match L\n");
      return(-1);
    }
    if(Nrand->ln != a->ln){
      printf("Col number of a does not match Nrand\n");
      return(-1);
    }
    if(Nrand->ln != b->ln){
      printf("Col number of b does not match Nrand\n");
      return(-1);
    }
    if(Nrand->ln != p->ln){
      printf("Col number of p does not match Nrand\n");
      return(-1);
    }
    if(y != NULL && Nrand->ln != y->ln){
      printf("Col number of y does not match Nrand\n");
      return(-1);
    }
    /* Quick return */
    morse_pmvn( Nrand, L, a, b, p, y, scalerIn, sequence, request );

    return MORSE_SUCCESS;

}

void morse_pmvn(MORSE_desc_t *Nrand, MORSE_desc_t *L, MORSE_desc_t *a,
    MORSE_desc_t *b, MORSE_desc_t *p, MORSE_desc_t *y, int &scalerIn,
    MORSE_sequence_t *sequence, MORSE_request_t *request)
{
    MORSE_context_t *morse;
    MORSE_option_t options;

    morse = morse_context_self();
    if (sequence->status != MORSE_SUCCESS)
        return;
    RUNTIME_options_init(&options, morse, sequence, request);
    int tempjm, tempkn, temprn;
    int ldLj, ldyr, ldabj;
    MORSE_Complex64_t alpha = 1.0;
    MORSE_Complex64_t beta = 1.0;
    for(int r = 0; r < L->mt; r++){
      if(r > 0){
        temprn = L->nb;
        ldyr = BLKLDD(y, r - 1);
        for(int j = r; j < L->mt; j++){
          tempjm = j == L->mt - 1? L->m - j * L->mb : L->mb;
          ldLj = BLKLDD(L, j);
          ldabj = BLKLDD(a, j);
          for(int k = 0; k < y->nt; k++){
            tempkn = k == y->nt - 1? y->n - k * y->nb : y->nb;
            MORSE_TASK_zgemm(&options, MorseNoTrans, MorseNoTrans, tempjm, 
                tempkn, temprn, L->mb, alpha, L, j, r - 1, ldLj,
                y, r - 1, k, ldyr, beta, a, j, k, ldabj);
            MORSE_TASK_zgemm(&options, MorseNoTrans, MorseNoTrans, tempjm, 
                tempkn, temprn, L->mb, alpha, L, j, r - 1, ldLj,
                y, r - 1, k, ldyr, beta, b, j, k, ldabj);
          }
        }
      }
      // continue here
      for(int k = 0; k < y->nt; k++){
        MORSE_TASK_mvndns(&options, )
      }

    }
}
