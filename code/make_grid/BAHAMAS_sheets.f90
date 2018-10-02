!Originally written by Alexander Mead
!Modified by Tilman Troester to allow for creation of sheets instead of cubic 
!grids.

PROGRAM create_sheets

  USE constants
  
  IMPLICIT NONE
  CHARACTER(len=256) :: outbase, inbase, outfile, Omega_m, Hubble, Length
  CHARACTER(len=256) :: infile_dm, infile_dm2, infile_gas, infile_stars, binning
  LOGICAL :: lexist_dm, lexist_dm2, lexist_gas, lexist_stars
  CHARACTER(len=256) :: mesh, sheets
  REAL, ALLOCATABLE :: x(:,:), mass(:), kT(:), nh(:)
  REAL, ALLOCATABLE :: d_dm(:,:,:,:), d_gas(:,:,:,:), d_stars(:,:,:,:), p_gas(:,:,:,:)
  INTEGER :: i, j, n, m, nk, n_sheet
  REAL :: total_box_mass, average_cell_mass, sn
  REAL :: Om_m, L, h
  REAL, ALLOCATABLE :: k(:), D2(:)
  INTEGER, ALLOCATABLE :: nbin(:)
  LOGICAL :: lexist, standard_gravity, dual_fluid_gravity, hydro

  !PARAMETERS etc.
  INTEGER :: ibin=2 !Choose CIC binning
  LOGICAL, PARAMETER :: do_pressure=.TRUE. !Compute the gas pressure field and associated spectra

  !TODO: Maybe make all fields output at the end of the code, easier debugging

  !Initialise the number-of-particles variable to zero
  !This is an array of size 3
  !np=0.

  !Initially set this flag to false
  !Set to true later if hydrodynamics are detected
  hydro=.FALSE.

  !Intially set all flags for simualtions types to be FALSE
  !One (and only one) should be set to true later by the code
  standard_gravity=.FALSE.
  dual_fluid_gravity=.FALSE.
  hydro=.FALSE.
 
  CALL get_command_argument(1,inbase)
  IF(inbase=='') STOP 'Specify input file base'

  CALL get_command_argument(2,Omega_m)
  IF(Omega_m=='') STOP 'Specifiy Omega_m'
  READ(Omega_m,*) Om_m
  
  CALL get_command_argument(3,Hubble)
  IF(Hubble=='') STOP 'Specify Hubble constant'
  READ(Hubble,*) h

  CALL get_command_argument(4,Length)
  IF(Length=='') STOP 'Specify simulation box size [Mpc/h]'
  READ(Length,*) L
  
  CALL get_command_argument(5,mesh)
  IF(mesh=='') STOP 'Specify mesh size'
  READ(mesh,*) m

  CALL get_command_argument(6,outbase)
  IF(outbase=='') STOP 'Specify output file base'

  CALL get_command_argument(7,sheets)
  IF(sheets=='') STOP 'Specify number of sheets'
  READ(sheets,*) n_sheet

  CALL get_command_argument(8,binning)
  IF(TRIM(binning)=='NGP') ibin=1
  IF(TRIM(binning)=='CIC') ibin=2

  WRITE(*,*)
  WRITE(*,*) 'OWLS_FIELD: field spectrum code'
  WRITE(*,*) '==============================='
  WRITE(*,*)

  WRITE(*,*) 'OWLS_FIELD: Omega_m:', Om_m
  WRITE(*,*) 'OWLS_FIELD: Hubble constant:', h
  WRITE(*,*) 'OWLS_FIELD: Box size [Mpc/h]:', L
  WRITE(*,*)

  !Calculate the total mass in the volume and the mean mass per cell
  total_box_mass=critical_density*Om_m*L**3
  average_cell_mass=total_box_mass/INT(m, kind=8)**3
  WRITE(*,*) 'OWLS_FIELD: Total box mass [Msun/h]:', total_box_mass
  WRITE(*,*) 'OWLS_FIELD: Number of mesh cells [cube root]:', m
  WRITE(*,*) 'OWLS_FIELD: Number of sheets (x 3 projections):', n_sheet
  WRITE(*,*) 'OWLS_FIELD: Average cell mass [Msun/h]:', average_cell_mass
  WRITE(*,*)
  
  !Allocate arrays for the fields
  ALLOCATE(d_dm(3,n_sheet,m,m),d_gas(3,n_sheet,m,m),d_stars(3,n_sheet,m,m))
  IF(do_pressure) ALLOCATE(p_gas(3,n_sheet,m,m))

  !Check for the existence of dm file
  infile_dm=TRIM(inbase)//'_dm.dat'
  INQUIRE(file=infile_dm, exist=lexist_dm)

  !Check for the existence of dm2 file
  infile_dm2=TRIM(inbase)//'_dm2.dat'
  INQUIRE(file=infile_dm2, exist=lexist_dm2)

  !Check for the existence of gas file
  infile_gas=TRIM(inbase)//'_gas.dat'
  INQUIRE(file=infile_gas, exist=lexist_gas)

  !Check for the existence of stars file
  infile_stars=TRIM(inbase)//'_stars.dat'
  INQUIRE(file=infile_stars, exist=lexist_stars)

  !Check what type of files are present and therefore what type of spectra to produce
  IF(lexist_dm .EQV. .FALSE.) THEN     
     WRITE(*,*) 'OWLS POWER: INFILE: ', TRIM(infile_dm)
     STOP 'OWLS_FIELD: Something is wrong, this input file does not exist'
  ELSE IF(lexist_dm .AND. (lexist_dm2 .EQV. .FALSE.) .AND. (lexist_gas .EQV. .FALSE.) .AND. (lexist_stars .EQV. .FALSE.)) THEN
     standard_gravity=.TRUE.
     WRITE(*,*) 'OWLS POWER: Standard gravity-only simulation'
  ELSE IF(lexist_dm2) THEN
     dual_fluid_gravity=.TRUE.
     WRITE(*,*) 'OWLS POWER: Dual fluid gravity-only simulation'
  ELSE IF(lexist_gas .AND. lexist_stars) THEN
     hydro=.TRUE.
     WRITE(*,*) 'OWLS POWER: Hydrodynamic simulation'
  ELSE
     STOP 'OWLS_FIELD: Some bizarre collection of input files are present'
  END IF
  
  !!

  !Read in dark-matter particles
  IF(lexist_dm) THEN

     !Read in the dm file
     CALL read_mccarthy(x,mass,n,infile_dm)
     CALL replace(x,n,L)

     !Save the number of particles
     !np(1)=n

     !Make the density field
     CALL particle_bin_sheets(x,n,L,mass,d_dm,m,n_sheet,ibin)
     d_dm=d_dm/average_cell_mass
     !total_box_mass2=SUM(mass)
     DEALLOCATE(x,mass)
     outfile=TRIM(outbase)//'_dm_sheets'
     CALL write_field_binary(d_dm,m,n_sheet,outfile)

  END IF

  !!

  !!

  !Read in second fluid dark-matter particles and label them as gas
  IF(lexist_dm2) THEN

     !Read in the dm file
     CALL read_mccarthy(x,mass,n,infile_dm2)
     CALL replace(x,n,L)

     !Save the number of particles
     !np(2)=n

     !Make the density field
     CALL particle_bin_sheets(x,n,L,mass,d_gas,m,n_sheet,ibin)
     d_gas=d_gas/average_cell_mass
     DEALLOCATE(x,mass)
     outfile=TRIM(outbase)//'_gas_sheets'
     CALL write_field_binary(d_gas,m,n_sheet,outfile)

  END IF

  !!

  !!

  !Read in the gas particles
  IF(lexist_gas) THEN

     !Read in the gas file
     IF(do_pressure) THEN
        CALL read_mccarthy_gas(x,mass,kT,nh,n,infile_gas)
     ELSE
        CALL read_mccarthy(x,mass,n,infile_gas)
     END IF
     CALL replace(x,n,L)

     !Save the number of particles
     !np(2)=n

     !Make the density and pressure fields
     CALL particle_bin_sheets(x,n,L,mass,d_gas,m,n_sheet,ibin)
     d_gas=d_gas/average_cell_mass
     outfile=TRIM(outbase)//'_gas_sheets'
     CALL write_field_binary(d_gas,m,n_sheet,outfile)
     
     IF(do_pressure) THEN
        CALL convert_kT_to_comoving_electron_pressure(kT,nh,mass,n,L,h,m)
        CALL particle_bin_sheets(x,n,L,kT,p_gas,m,n_sheet,ibin)        
        DEALLOCATE(x,mass,kT,nh)
        outfile=TRIM(outbase)//'_pressure_sheets'
        CALL write_field_binary(p_gas,m,n_sheet,outfile)
     ELSE
        DEALLOCATE(x,mass)
     END IF   

  END IF

  !!

  !!

  !Read in the star particles
  IF(lexist_stars) THEN

     !Read in the stars file
     CALL read_mccarthy(x,mass,n,infile_stars)
     CALL replace(x,n,L)

     !Save the number of particles
     !np(3)=n

     !Make the denstiy field
     CALL particle_bin_sheets(x,n,L,mass,d_stars,m,n_sheet,ibin)
     d_stars=d_stars/average_cell_mass
     DEALLOCATE(x,mass)
     outfile=TRIM(outbase)//'_stars_sheets'
     CALL write_field_binary(d_stars,m,n_sheet,outfile)

  END IF

  DEALLOCATE(d_dm,d_gas,d_stars)
  IF(ALLOCATED(p_gas)) DEALLOCATE(p_gas)


  CONTAINS

   SUBROUTINE read_mccarthy(x,m,n,infile)

    IMPLICIT NONE
    CHARACTER(len=*), INTENT(IN) :: infile
    REAL, ALLOCATABLE, INTENT(OUT) :: x(:,:), m(:)
    INTEGER, INTENT(OUT) :: n
    REAL, PARAMETER :: mfac=1e10

    WRITE(*,*) 'READ_MCCARTHY: Reading in binary file: ', TRIM(infile)

    OPEN(7,file=infile,form='unformatted',access='stream',status='old')
    READ(7) n
    CLOSE(7)

    ! In case the array is empty, but actually Ian has n=1 set (e.g., UVB_stars)
    IF(n==1) THEN
       n=0
    END IF

    WRITE(*,*) 'READ_MCCARTHY: Particle number:', n
    WRITE(*,*) 'READ_MCCARTHY: Which is ~', NINT(n**(1./3.)), 'cubed.'

    ALLOCATE(x(3,n),m(n))

    IF(n .NE. 0) THEN

       ! Need to read in 'n' again with stream access
       OPEN(7,file=infile,form='unformatted',access='stream',status='old')
       READ(7) n
       READ(7) m
       READ(7) x
       CLOSE(7)

       m=m*mfac
       
       WRITE(*,*) 'READ_MCCARTHY: Minimum particle mass [Msun/h]:', MINVAL(m)
       WRITE(*,*) 'READ_MCCARTHY: Maximum particle mass [Msun/h]:', MAXVAL(m)
       WRITE(*,*) 'READ_MCCARTHY: Total particle mass [Msun/h]:', SUM(m)
       WRITE(*,*) 'READ_MCCARTHY: Minimum x coordinate [Mpc/h]:', MINVAL(x(1,:))
       WRITE(*,*) 'READ_MCCARTHY: Minimum x coordinate [Mpc/h]:', MAXVAL(x(1,:))
       WRITE(*,*) 'READ_MCCARTHY: Minimum y coordinate [Mpc/h]:', MINVAL(x(2,:))
       WRITE(*,*) 'READ_MCCARTHY: Minimum y coordinate [Mpc/h]:', MAXVAL(x(2,:))
       WRITE(*,*) 'READ_MCCARTHY: Minimum z coordinate [Mpc/h]:', MINVAL(x(3,:))
       WRITE(*,*) 'READ_MCCARTHY: Minimum z coordinate [Mpc/h]:', MAXVAL(x(3,:))
       WRITE(*,*) 'READ_MCCARTHY: Finished reading in file'

    END IF

    WRITE(*,*)

  END SUBROUTINE read_mccarthy

  SUBROUTINE read_mccarthy_gas(x,m,kT,nh,n,infile)

    USE constants
    IMPLICIT NONE
    CHARACTER(len=*), INTENT(IN) :: infile
    REAL, ALLOCATABLE, INTENT(OUT) :: x(:,:), m(:), nh(:), kT(:)
    REAL, ALLOCATABLE :: ep(:)
    INTEGER, INTENT(OUT) :: n
    
    REAL, PARAMETER :: mfac=1e10 ! Convert mass to Solar masses
    REAL, PARAMETER :: eV_erg=eV*1e7 ! eV in ergs

    ! Read in the binary file
    WRITE(*,*) 'READ_MCCARTHY_GAS: Reading in binary file: ', TRIM(infile)
    OPEN(7,file=infile,form='unformatted',access='stream',status='old')
    READ(7) n
    CLOSE(7)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Particle number:', n
    WRITE(*,*) 'READ_MCCARTHY_GAS: Which is ~', NINT(n**(1./3.)), 'cubed.'

    ! Allocate arrays for quantities in the file
    ALLOCATE(x(3,n),m(n),ep(n),nh(n))
    
    ! Need to read in 'n' again with stream access
    OPEN(7,file=infile,form='unformatted',access='stream',status='old')
    READ(7) n
    READ(7) m
    READ(7) x
    READ(7) ep ! physical electron pressure for the particle in erg/cm^3
    READ(7) nh ! hydrogen number density for the partcle in /cm^3
    CLOSE(7)

    ! Convert masses into Solar masses
    m=m*mfac

    WRITE(*,*) 'READ_MCCARTHY_GAS: Calculating kT from physical electron pressure'
    WRITE(*,*) 'READ_MCCARTHY_GAS: Note that the electron pressure is *not* comoving'
    WRITE(*,*) 'READ_MCCARTHY_GAS: Using numbers appropriate for BAHAMAS'
    WRITE(*,*) 'READ_MCCARTHY_GAS: YH:', fh
    WRITE(*,*) 'READ_MCCARTHY_GAS: mu_H:', mu
    WRITE(*,*) 'READ_MCCARTHY_GAS: Xe:', Xe
    WRITE(*,*) 'READ_MCCARTHY_GAS: Xi:', Xi
    
    ! Convert the physical electron pressure [erg/cm^3] and hydrogen density [#/cm^3] into kT
    ! Units of kT will be [erg]
    ! This is the temperature of gas particles (equal for all species)
    ! Temperature is neither comoving nor physical
    ALLOCATE(kT(n))
    kT=((Xe+Xi)/Xe)*(ep/nh)*mu*fh

    ! Convert internal energy from erg to eV
    kT=kT/eV_erg

    ! Deallocate the physical electron pressure array
    DEALLOCATE(ep)

    ! Write information to the screen
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum particle mass [Msun/h]:', MINVAL(m)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Maximum particle mass [Msun/h]:', MAXVAL(m)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum x coordinate [Mpc/h]:', MINVAL(x(1,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum x coordinate [Mpc/h]:', MAXVAL(x(1,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum y coordinate [Mpc/h]:', MINVAL(x(2,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum y coordinate [Mpc/h]:', MAXVAL(x(2,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum z coordinate [Mpc/h]:', MINVAL(x(3,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum z coordinate [Mpc/h]:', MAXVAL(x(3,:))
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum internal energy [eV]:', MINVAL(kT)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Maximum internal energy [eV]:', MAXVAL(kT)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Minimum hydrogen number density [cm^-3]:', MINVAL(nh)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Maximum hydrogen number density [cm^-3]:', MAXVAL(nh)
    WRITE(*,*) 'READ_MCCARTHY_GAS: Finished reading in file'
    WRITE(*,*)

  END SUBROUTINE read_mccarthy_gas

  SUBROUTINE convert_kT_to_comoving_electron_pressure(kT,nh,mass,n,L,h,m)

    ! kT is particle internal energy input in units of eV, it is output in units of eV/cm^3
    ! nh is hydrogen number density in units /cm^3
    ! mass is particle mass in units of msun
    ! n is the total number of particles
    ! L is the box size in units of Mpc/h
    ! h is the dimensionless hubble parameter
    ! m is the mesh size onto which the pressure will be binned
    USE constants
    IMPLICIT NONE
    REAL, INTENT(INOUT) :: kT(n)
    REAL, INTENT(IN) :: mass(n), nh(n), L, h
    INTEGER, INTENT(IN) :: n, m
    REAL :: V
    DOUBLE PRECISION :: units, kT_dble(n)
    
    LOGICAL, PARAMETER :: apply_nh_cut=.TRUE. ! Apply a cut in hydrogen density
    REAL, PARAMETER :: nh_cut=0.1 ! Cut in the hydrogen number density [cm^-3] gas denser than this is not ionised

    ! Exclude gas that is sufficiently dense to not be ionised and be forming stars
    IF(apply_nh_cut) CALL exclude_nh(nh_cut,kT,nh,n)

    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Converting kT to comoving electron pressure'
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Using numbers appropriate for BAHAMAS'
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Note that this is COMOVING'
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Y_H:', fh
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: mu_H:', mu
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Xe:', Xe
    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Xi:', Xi

    ! Use double precision because all the constants are dreadful 
    kT_dble=kT! [eV]

    ! Convert to particle internal energy that needs to be mapped to grid
    kT_dble=kT_dble*(mass/mu)*Xe/(Xe+Xi)! [eV*Msun]

    ! Comoving cell volume
    V=(L/REAL(m))**3! [(Mpc/h)^3]
    V=V/h**3 ! remove h factors [Mpc^3]

    ! This is now comoving electron pressure
    kT_dble=kT_dble/V! [Msun*eV/Mpc^3]

    ! Convert units of comoving electron pressure
    ! Note that there are no h factors here
    units=msun
    units=units/mp
    units=units/(Mpc/cm)
    units=units/(Mpc/cm)
    units=units/(Mpc/cm)
    kT_dble=kT_dble*units! [eV/cm^3]

    ! Go back to single precision
    kT=REAL(kT_dble)! [eV/cm^3]

    WRITE(*,*) 'CONVERT_KT_TO_ELECTRON_PRESSURE: Done'
    WRITE(*,*)

  END SUBROUTINE convert_kT_to_comoving_electron_pressure

  SUBROUTINE exclude_nh(nhcut,ep,nh,n)

    ! Set the electron pressure to zero of any particle that has nh > nhcut
    IMPLICIT NONE
    REAL, INTENT(IN) :: nhcut, nh(n)
    REAL, INTENT(INOUT) :: ep(n)
    INTEGER, INTENT(IN) :: n
    INTEGER :: i

    DO i=1,n
       IF(nh(i)>nhcut) ep(i)=0.
    END DO
    
  END SUBROUTINE exclude_nh

  SUBROUTINE replace(x,n,L)

    ! Ensures/enforces periodicity by cycling particles round that may have strayed
    ! This forces all particles to be 0<=x<L, so they cannot be exactly at x=L
    IMPLICIT NONE
    REAL, INTENT(INOUT) :: x(3,n)
    REAL, INTENT(IN) :: L
    INTEGER :: i, j
    INTEGER, INTENT(IN) :: n

    DO i=1,n
       DO j=1,3
          IF(x(j,i)>=L) x(j,i)=x(j,i)-L
          IF(x(j,i)<0.) x(j,i)=x(j,i)+L
       END DO
    END DO

  END SUBROUTINE replace

  SUBROUTINE particle_bin(x,n,L,w,d,m,ibin)

    !Bin particle properties onto a mesh, summing as you go
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m
    INTEGER, INTENT(INOUT) :: ibin
    REAL, INTENT(INOUT) :: d(m,m,m)
    REAL, INTENT(IN) :: x(3,n), L, w(n)

    IF(ibin==-1) THEN
       WRITE(*,*) 'Choose binning strategy'
       WRITE(*,*) '1 - NGP'
       WRITE(*,*) '2 - CIC'
       READ(*,*) ibin
       WRITE(*,*)
    END IF

    IF(ibin==1) THEN
       CALL NGP(x,n,L,w,d,m)
    ELSE IF(ibin==2) THEN
       CALL CIC(x,n,L,w,d,m)
    ELSE
       STOP 'PARTICLE_BIN: Error, ibin not specified correctly'
    END IF

  END SUBROUTINE particle_bin

  SUBROUTINE particle_bin_sheets(x,n,L,w,d,m,n_sheet,ibin)

    !Bin particle properties onto a mesh, summing as you go
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m, n_sheet
    INTEGER, INTENT(INOUT) :: ibin
    REAL, INTENT(INOUT) :: d(3,n_sheet,m,m)
    REAL, INTENT(IN) :: x(3,n), L, w(n)

    IF(ibin==-1) THEN
       WRITE(*,*) 'Choose binning strategy'
       WRITE(*,*) '1 - NGP'
       WRITE(*,*) '2 - CIC'
       READ(*,*) ibin
       WRITE(*,*)
    END IF

    IF(ibin==1) THEN
       CALL NGP_sheets(x,n,L,w,d,m,n_sheet)
    ELSE IF(ibin==2) THEN
       CALL CIC_sheets(x,n,L,w,d,m,n_sheet)
    ELSE
       STOP 'PARTICLE_BIN_SHEETS: Error, ibin not specified correctly'
    END IF

  END SUBROUTINE particle_bin_sheets

  SUBROUTINE NGP(x,n,L,w,d,m)

    ! Nearest-grid-point binning routine
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m
    REAL, INTENT(OUT) :: d(m,m,m)
    REAL, INTENT(IN) :: x(3,n), w(n), L
    INTEGER :: i, ix, iy, iz

    WRITE(*,*) 'NGP: Binning particles and creating field'
    WRITE(*,*) 'NGP: Cells:', m

    ! Set array to zero explicitly
    d=0.

    DO i=1,n

       ! Get integer coordiante of the cell
       ix=NGP_cell(x(1,i),L,m)
       iy=NGP_cell(x(2,i),L,m)
       iz=NGP_cell(x(3,i),L,m)

       ! Bin
       d(ix,iy,iz)=d(ix,iy,iz)+w(i)

    END DO

    WRITE(*,*) 'NGP: Minimum:', MINVAL(REAL(d))
    WRITE(*,*) 'NGP: Maximum:', MAXVAL(REAL(d))
    WRITE(*,*) 'NGP: Binning complete'
    WRITE(*,*)

  END SUBROUTINE NGP

  SUBROUTINE NGP_sheets(x,n,L,w,d,m,n_sheet)

    ! Nearest-grid-point binning routine
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m, n_sheet
    REAL, INTENT(OUT) :: d(3,n_sheet,m,m)
    REAL, INTENT(IN) :: x(3,n), w(n), L
    INTEGER :: i, ix, iy, iz, isx, isy, isz

    WRITE(*,*) 'NGP: Binning particles and creating field'
    WRITE(*,*) 'NGP: Cells, sheets:', m, n_sheet

    ! Set array to zero explicitly
    d=0.

    DO i=1,n

       ! Get integer coordiante of the cell
       ix=NGP_cell(x(1,i),L,m)
       iy=NGP_cell(x(2,i),L,m)
       iz=NGP_cell(x(3,i),L,m)

       isx=CEILING(REAL(ix)/m*n_sheet)
       isy=CEILING(REAL(iy)/m*n_sheet)
       isz=CEILING(REAL(iz)/m*n_sheet)

       ! Bin
       d(1,isx,iy,iz) = d(1,isx,iy,iz) + w(i)
       d(2,isy,ix,iz) = d(2,isy,ix,iz) + w(i)
       d(3,isz,ix,iy) = d(3,isz,ix,iy) + w(i)

    END DO

    WRITE(*,*) 'NGP: Minimum:', MINVAL(REAL(d))
    WRITE(*,*) 'NGP: Maximum:', MAXVAL(REAL(d))
    WRITE(*,*) 'NGP: Binning complete'
    WRITE(*,*)

  END SUBROUTINE NGP_sheets

  SUBROUTINE CIC(x,n,L,w,d,m)

    ! Cloud-in-cell binning routine
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m
    REAL, INTENT(IN) :: x(3,n), L, w(n)
    REAL, INTENT(OUT) :: d(m,m,m)
    INTEGER :: ix, iy, iz, ixn, iyn, izn
    INTEGER :: i
    REAL :: dx, dy, dz

    WRITE(*,*) 'CIC: Binning particles and creating field'
    WRITE(*,*) 'CIC: Cells:', m

    ! Set array to zero explicitly
    d=0.

    DO i=1,n

       ! Integer coordinates of the cell the particle is in
       ix=NGP_cell(x(1,i),L,m)
       iy=NGP_cell(x(2,i),L,m)
       iz=NGP_cell(x(3,i),L,m)

       ! dx, dy, dz in box units
       dx=(x(1,i)/L)*REAL(m)-(REAL(ix)-0.5)
       dy=(x(2,i)/L)*REAL(m)-(REAL(iy)-0.5)
       dz=(x(3,i)/L)*REAL(m)-(REAL(iz)-0.5)

       IF(dx>=0.) THEN
          ixn=ix+1
          IF(ixn>m) ixn=1
       ELSE
          ixn=ix-1
          dx=-dx  
          IF(ixn<1) ixn=m    
       END IF

       IF(dy>=0.) THEN
          iyn=iy+1
          IF(iyn>m) iyn=1
       ELSE
          iyn=iy-1
          dy=-dy
          IF(iyn<1) iyn=m
       END IF

       IF(dz>=0.) THEN
          izn=iz+1
          IF(izn>m) izn=1
       ELSE
          izn=iz-1
          dz=-dz
          IF(izn<1) izn=m
       END IF

       ! Do the CIC binning
       d(ix,iy,iz)=d(ix,iy,iz)+(1.-dx)*(1.-dy)*(1.-dz)*w(i)
       d(ix,iy,izn)=d(ix,iy,izn)+(1.-dx)*(1.-dy)*dz*w(i)
       d(ix,iyn,iz)=d(ix,iyn,iz)+(1.-dx)*dy*(1.-dz)*w(i)
       d(ixn,iy,iz)=d(ixn,iy,iz)+dx*(1.-dy)*(1.-dz)*w(i)
       d(ix,iyn,izn)=d(ix,iyn,izn)+(1.-dx)*dy*dz*w(i)
       d(ixn,iyn,iz)=d(ixn,iyn,iz)+dx*dy*(1.-dz)*w(i)
       d(ixn,iy,izn)=d(ixn,iy,izn)+dx*(1.-dy)*dz*w(i)
       d(ixn,iyn,izn)=d(ixn,iyn,izn)+dx*dy*dz*w(i)

    END DO

    ! Write out some statistics to screen
    WRITE(*,*) 'CIC: Minimum:', MINVAL(REAL(d))
    WRITE(*,*) 'CIC: Maximum:', MAXVAL(REAL(d))
    WRITE(*,*) 'CIC: Binning complete'
    WRITE(*,*)

  END SUBROUTINE CIC

  SUBROUTINE CIC_sheets(x,n,L,w,d,m,n_sheet)

    ! Cloud-in-cell binning routine
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n, m, n_sheet
    REAL, INTENT(IN) :: x(3,n), L, w(n)
    REAL, INTENT(OUT) :: d(3,n_sheet,m,m)
    INTEGER :: ix, iy, iz, ixn, iyn, izn, isx, isy, isz, isxn, isyn, iszn
    INTEGER :: i
    REAL :: dx, dy, dz

    WRITE(*,*) 'CIC: Binning particles and creating field'
    WRITE(*,*) 'CIC: Cells, sheets:', m, n_sheet

    ! Set array to zero explicitly
    d=0.

    DO i=1,n

       ! Integer coordinates of the cell the particle is in
       ix=NGP_cell(x(1,i),L,m)
       iy=NGP_cell(x(2,i),L,m)
       iz=NGP_cell(x(3,i),L,m)

       ! dx, dy, dz in box units
       dx=(x(1,i)/L)*REAL(m)-(REAL(ix)-0.5)
       dy=(x(2,i)/L)*REAL(m)-(REAL(iy)-0.5)
       dz=(x(3,i)/L)*REAL(m)-(REAL(iz)-0.5)

       IF(dx>=0.) THEN
          ixn=ix+1
          IF(ixn>m) ixn=1
       ELSE
          ixn=ix-1
          dx=-dx  
          IF(ixn<1) ixn=m    
       END IF

       IF(dy>=0.) THEN
          iyn=iy+1
          IF(iyn>m) iyn=1
       ELSE
          iyn=iy-1
          dy=-dy
          IF(iyn<1) iyn=m
       END IF

       IF(dz>=0.) THEN
          izn=iz+1
          IF(izn>m) izn=1
       ELSE
          izn=iz-1
          dz=-dz
          IF(izn<1) izn=m
       END IF

       isx=CEILING(REAL(ix)/m*n_sheet)
       isy=CEILING(REAL(iy)/m*n_sheet)
       isz=CEILING(REAL(iz)/m*n_sheet)

       isxn=CEILING(REAL(ixn)/m*n_sheet)
       isyn=CEILING(REAL(iyn)/m*n_sheet)
       iszn=CEILING(REAL(izn)/m*n_sheet)

       ! Do the CIC binning
       d(1,isx,iy,iz)=d(1,isx,iy,iz)+(1.-dx)*(1.-dy)*(1.-dz)*w(i)
       d(1,isx,iy,izn)=d(1,isx,iy,izn)+(1.-dx)*(1.-dy)*dz*w(i)
       d(1,isx,iyn,iz)=d(1,isx,iyn,iz)+(1.-dx)*dy*(1.-dz)*w(i)
       d(1,isxn,iy,iz)=d(1,isxn,iy,iz)+dx*(1.-dy)*(1.-dz)*w(i)
       d(1,isx,iyn,izn)=d(1,isx,iyn,izn)+(1.-dx)*dy*dz*w(i)
       d(1,isxn,iyn,iz)=d(1,isxn,iyn,iz)+dx*dy*(1.-dz)*w(i)
       d(1,isxn,iy,izn)=d(1,isxn,iy,izn)+dx*(1.-dy)*dz*w(i)
       d(1,isxn,iyn,izn)=d(1,isxn,iyn,izn)+dx*dy*dz*w(i)

       d(2,isy,ix,iz)=d(2,isy,ix,iz)+(1.-dx)*(1.-dy)*(1.-dz)*w(i)
       d(2,isy,ix,izn)=d(2,isy,ix,izn)+(1.-dx)*(1.-dy)*dz*w(i)
       d(2,isy,ixn,iz)=d(2,isy,ixn,iz)+(1.-dx)*dy*(1.-dz)*w(i)
       d(2,isyn,ix,iz)=d(2,isyn,ix,iz)+dx*(1.-dy)*(1.-dz)*w(i)
       d(2,isy,ixn,izn)=d(2,isy,ixn,izn)+(1.-dx)*dy*dz*w(i)
       d(2,isyn,ixn,iz)=d(2,isyn,ixn,iz)+dx*dy*(1.-dz)*w(i)
       d(2,isyn,ix,izn)=d(2,isyn,ix,izn)+dx*(1.-dy)*dz*w(i)
       d(2,isyn,ixn,izn)=d(2,isyn,ixn,izn)+dx*dy*dz*w(i)

       d(3,isz,ix,iy)=d(3,isz,ix,iy)+(1.-dx)*(1.-dy)*(1.-dz)*w(i)
       d(3,isz,ix,iyn)=d(3,isz,ix,iyn)+(1.-dx)*(1.-dy)*dz*w(i)
       d(3,isz,ixn,iy)=d(3,isz,ixn,iy)+(1.-dx)*dy*(1.-dz)*w(i)
       d(3,iszn,ix,iy)=d(3,iszn,ix,iy)+dx*(1.-dy)*(1.-dz)*w(i)
       d(3,isz,ixn,iyn)=d(3,isz,ixn,iyn)+(1.-dx)*dy*dz*w(i)
       d(3,iszn,ixn,iy)=d(3,iszn,ixn,iy)+dx*dy*(1.-dz)*w(i)
       d(3,iszn,ix,iyn)=d(3,iszn,ix,iyn)+dx*(1.-dy)*dz*w(i)
       d(3,iszn,ixn,iyn)=d(3,iszn,ixn,iyn)+dx*dy*dz*w(i)

    END DO

    ! Write out some statistics to screen
    WRITE(*,*) 'CIC: Minimum:', MINVAL(REAL(d))
    WRITE(*,*) 'CIC: Maximum:', MAXVAL(REAL(d))
    WRITE(*,*) 'CIC: Binning complete'
    WRITE(*,*)

  END SUBROUTINE CIC_sheets

  SUBROUTINE write_field_binary(d,m,n_sheet,outfile)

    ! Write out a binary 'field' file
    IMPLICIT NONE    
    REAL, INTENT(IN) :: d(3,n_sheet,m,m)
    INTEGER, INTENT(IN) :: m, n_sheet
    CHARACTER(len=*), INTENT(IN) :: outfile
    INTEGER :: unit

    WRITE(*,*) 'WRITE_FIELD_BINARY: Binary output: ', TRIM(outfile)
    WRITE(*,*) 'WRITE_FIELD_BINARY: Mesh size:', m
    WRITE(*,*) 'WRITE_FIELD_BINARY: Number of sheets:', n_sheet
    WRITE(*,*) 'WRITE_FIELD_BINARY: Minval:', MINVAL(d)
    WRITE(*,*) 'WRITE_FIELD_BINARY: Maxval:', MAXVAL(d)
    WRITE(*,*) 'WRITE_FIELD_BINARY: Using new version with access=stream'
    OPEN(newunit=unit,file=outfile,form='unformatted',access='stream',status='replace')
    WRITE(unit) d
    CLOSE(unit)
    WRITE(*,*) 'WRITE_3D_FIELD_BINARY: Done'
    WRITE(*,*)

  END SUBROUTINE write_field_binary

  INTEGER FUNCTION NGP_cell(x,L,m)

    ! Find the integer coordinates of the cell that coordinate x is in
    IMPLICIT NONE
    REAL, INTENT(IN) :: x ! Particle position
    REAL, INTENT(IN) :: L ! Box size
    INTEGER, INTENT(IN) :: m ! Number of mesh cells in grid

    IF(x==0.) THEN
       ! Catch this edge case
       NGP_cell=1
    ELSE
       NGP_cell=CEILING(x*REAL(m)/L)
    END IF

    IF(NGP_cell<1 .OR. NGP_cell>m) THEN
       WRITE(*,*) 'NGP_CELL: Particle position [Mpc/h]:', x
       WRITE(*,*) 'NGP_CELL: Box size [Mpc/h]:', L
       WRITE(*,*) 'NGP_CELL: Mesh size:', m 
       WRITE(*,*) 'NGP_CELL: Assigned cell:', NGP_cell
       STOP 'NGP_CELL: Error, the assigned cell position is outside the mesh'
    END IF

  END FUNCTION NGP_cell
  
END PROGRAM create_sheets
