
type input_file

  use sizes
  logical :: tinker, pdb, xyz, moldy
  integer :: nlines
  character(len=strl), allocatable :: line(:)
  character(len=strl), allocatable :: keyword(:,:)

end type input_file
