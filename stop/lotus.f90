program cylinder3d
  use fluidMod,   only: fluid
  use fieldMod,   only: field
  use vectorMod
  use bodyMod,    only: body
  use mympiMod
  use gridMod
  use imageMod,   only: display
  use ioMod,      only:write_vtk
  use geom_shape
  use uMod,      only: all_profile,ptheta,avrgField
  implicit none

! -- Physical parameters
  real,parameter  :: Re = 3900, St = 1.5
  integer         :: ndims = 2
  real            :: z_D	                                            ! span based on diameter
!
! -- numerical parameters
  real,parameter  :: D = 8           	                              ! resolution (pnts per diameter)
  real,parameter  :: nu = D/Re, f = St/D
! -- MPI utils
  integer         :: n(3)             	                              ! number of points
  integer         :: b(3) = [4,2,1]                                   ! blocks
  real            :: m(3)                                             ! number of points per block
  logical         :: root                                             ! root processor
  logical         :: kill=.false., p(3) = (/.FALSE.,.FALSE.,.TRUE./)
!
! -- utils
  real,parameter  :: dtPrint = 0.1*D                                  ! print rate
  real,parameter  :: angle = pi/5.                                   ! sample angle
  real            :: finish = 150.0*D
!
! -- Variables
  type(fluid)     :: flow
  type(field)     :: ave_p0
  type(vfield)    :: ave_v0
  type(body)      :: geom
  real            :: dt, t, pforce(3), vforce(3), cf_f(3), cf_s(3)
!
! -- Initialize MPI (if MPI is OFF, b is set to 1)
  call init_mympi(ndims,set_blocks=b(1:ndims),set_periodic=p(1:ndims))
  root = mympi_rank()==0
!
! -- Print MPI stuff
  if(root) print *, 'Blocks: ', b
  if(root) print *, 'Setting up the grid, body and fluid'
  if(root) print *,'-----------------------------------'
! -- Set-up grid
  if(ndims==3) then
    z_D = 3.141593
  else
    z_D = 0.
  end if

  m = D*[16.,8.,z_D]
  n = composite(m, prnt=root)
  call xg(1)%stretch(n(1), -14.*D, -1*D, 3.5*D, 26.*D, h_max=8., prnt=root)
  call xg(2)%stretch(n(2), -10*D, -1.6*D, 1.6*D, 10*D, prnt=root)
  if(ndims==3) xg(3)%h = 2.5

! -- Initialize cylinder
  geom = cylinder(axis=3,radius=0.5*D,center=0.).map.init_rigid(2,heave)

! -- Initialize fluid and other terms
  call flow%init(n/b, geom, V=[1.,0.,0.], nu=nu)
  if(root) print *,'Init complete. Starting time update loop'
  if(root) print *,'-----------------------------------'
  if(root) print *,' -t- , -dt- , -t_left- '

  finish = finish+flow%time ! For flows that are restarted
  time_loop: do while (flow%time<finish)
    dt = flow%dt
    call geom%update(flow%time+flow%dt)
    call flow%update(geom)
    t = flow%time

    if (t==150.*D) call flow%write()

    if((mod(t,15*D)<dt).and.(root)) call system('python3 converged.py 15 "px" "t, dt, px, py, pz, vx, vy, vz, v2x, v2y, v2z"')

    ave_v0 = flow%velocity%average()
    ave_p0 = flow%pressure%average()

    pforce = -2.*geom%pforce(flow%pressure)/(D*n(3)*xg(3)%h)
    vforce = 2.*nu*geom%vforce(flow%velocity)/(D*n(3)*xg(3)%h)
    cf_f = 2.*nu*geom%vforce_f(flow%velocity)/(D*n(3)*xg(3)%h)
    cf_s = 2.*nu*geom%vforce_s(flow%velocity)/(D*n(3)*xg(3)%h)
    
    ! Track simulation current status
    if((mod(t,0.5*D)<dt).and.(root)) print "('Time:',f15.3,'. Time remaining:',f15.3,3f12.6)",t/D,finish/D-t/D
    
  ! Write some viscous forces to see how first order method converges
    if(root) then
      write(9,'(f10.4,f8.4,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8)') t/D,dt,pforce,cf_f,cf_s,vforce
      flush(9)
    end if
  !
  ! -- write some profiles of the spanwise average
    call ptheta(flow%pressure, D, angle)
    call all_profile(ave_v0%e(1), 256, 3.*D, angle)
    call all_profile(ave_v0%e(2), 256, 3.*D, angle)

    ! Check if .kill, end the time_loop
    inquire(file=".kill", exist=kill)
    if (kill) exit time_loop
    
  end do time_loop

  ! Output
  call flow%write()
  call flow%write(average=.true.)

  call mympi_end
  
contains
!
! -- Define a wobble to trip bifurcation
real(8) pure function heave(t)
  real(8),intent(in) :: t
    if (t<0.5*D) then
      heave = 0.02*D*sin(2*pi*f*t)
    else
      heave = 0
    end if
  end function heave

end program cylinder3d
