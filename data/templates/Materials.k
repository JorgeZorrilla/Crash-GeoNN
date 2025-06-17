$# LS-DYNA Keyword file created by LS-PrePost(R) V4.10.11-27Mar2024
$# Created on Jun-17-2025 (19:18:33)
*KEYWORD
*TITLE
$#                                                                         title
LS-DYNA keyword deck by LS-PrePost
*MAT_RIGID_TITLE
Acero-Rigido
$#     mid        ro         e        pr         n    couple         m     alias
         17.85000E-9   2100000       0.0       0.0       0.0       0.0          
$#     cmo      con1      con2
       0.0         0         0
$#lco or a1        a2        a3        v1        v2        v3
       0.0       0.0       0.0       0.0       0.0       0.0
*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE
UHSS Material PLP
$#     mid        ro         e        pr      sigy      etan      fail      tdel
         27.85000E-9  210000.0       0.3    2100.0    5000.01.00000E21       0.0
$#       c         p      lcss      lcsr        vp
       0.0       0.0         0         0       0.0
$#    eps1      eps2      eps3      eps4      eps5      eps6      eps7      eps8
       0.0      0.01      0.02       0.1      0.15       0.0       0.0       0.0
$#     es1       es2       es3       es4       es5       es6       es7       es8
    2100.0    2200.0    2300.0    2400.0    2500.0       0.0       0.0       0.0
*MAT_JOHNSON_COOK_TITLE
UHSS Material Johnson-Cook
$#     mid        ro         g         e        pr       dtf        vp    rateop
         37.85000E-9   81500.0  210000.0       0.3       0.0       0.0       1.0
$#       a         b         n         c         m        tm        tr      epso
    1000.0     800.0       0.5     0.025      1.03    1800.0     300.0       1.0
$#      cp        pc     spall        it        d1        d2        d3        d4
     477.0       7.8       2.0       0.0      -0.5       1.5    -0.002      0.61
$#      d5c2/p/xnp/d      erod     efmin    numint         k      eps1
       0.0       1.0      0.011.00000E-6       0.0       0.0       0.0
*END
