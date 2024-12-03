# Orthonormal-Basis-Calculator
Calculates an orthonormal basis for a set of vectors using multi-threading techniques.

The purpose of this program is as an exercise in utilizing parallel processing to improve performance. The function of the program (calculating an orthonormal basis) was chosen because it offers straightforward opportunities to implement multi-threading.

This program takes a matrix as an argument, where the columns of the matrix form a linearly independent basis for the column space of the matrix.
Then it uses the matrix to calculate an orthonormal basis the column space of the same matrix.

Input format is:

    row col
    0 0 0 .. 0
    0 0 0 .. 0
    0 0 0 .. 0
    : : :    :
    0 0 0 .. 0
    
If the vectors (columns) are linearly dependent, then the orthonormal basis will have fewer vectors than the original matrix
(because linear dependence means that one or more vectors can be written as linear combinations of the others, and are therefore redundant).
If they are linearly independent then the orthonormal basis will have the same number of vectors.

*/
