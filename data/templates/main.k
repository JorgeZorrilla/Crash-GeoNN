*KEYWORD  
*TITLE
Impact Simulation
*INCLUDE
C:\Users\jorge\Documents\M2i\Crash-GeoNN\data\templates\Materials.k 
*INCLUDE
C:\Users\jorge\Documents\M2i\Crash-GeoNN\data\impactor\ImpactorKeyword.k 
*INCLUDE
${GEOMETRY_KEYWORD_PATH}
*PART
$#                                                                         title
B-Pillar
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         2         2         2         0         0         0         0         0
*SECTION_SHELL_TITLE
B-Pillar-Section
$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp
         2         2       1.0         2       1.0         0         0         1
$#      t1        t2        t3        t4      nloc     marea      idof    edgset
       1.5       1.5       1.5       1.5       0.0       0.0       0.0         0
*INITIAL_VELOCITY_GENERATION
$#      id      styp     omega        vx        vy        vz     ivatn      icid
         1         2       0.0       0.0    9160.0       0.0         0         0
$#      xc        yc        zc        nx        ny        nz     phase    irigid
       0.0       0.0       0.0       0.0       0.0       0.0         0         0
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_ID
$#     cid                                                                 title
         1Contact-B-Pillar-Impactor
$#   surfa     surfb  surfatyp  surfbtyp   saboxid   sbboxid      sapr      sbpr
         1         2         3         3         0         0         0         0
$#      fs        fd        dc        vc       vdc    penchk        bt        dt
       0.2       0.1       0.0       0.0       0.0         0       0.01.00000E20
$#    sfsa      sfsb      sast      sbst     sfsat     sfsbt       fsf       vsf
       1.0       1.0       0.0       0.0       1.0       1.0       1.0       1.0
*CONTROL_ENERGY
$#    hgen      rwen    slnten     rylen     irgen     maten     drlen     disen
         2         2         2         2         2         1         1         1
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol
     0.015         0       0.0       0.01.000000E8         0
*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt
1.50000E-4         0         0         1
*DATABASE_MATSUM
$#      dt    binary      lcur     ioopt
1.50000E-4         0         0         1
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
1.50000E-4         0         0         0         0
*END