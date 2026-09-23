"""Fixed, zero-parameter physics operator for PC-STGNN.

See ``PC-STGNN/01_SPEC.md`` for the frozen specification.  Nothing in this
package is ever learned, and nothing in it reads a network output: the residual
is a function of the input window and the signature table alone
(``01_SPEC.md`` §5).
"""
