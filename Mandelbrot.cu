#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkUniformGrid.h>
#include <vtkLookupTable.h>
#include <vtkImageMapToColors.h>
#include <vtkImageActor.h>
#include <vtkImageMapper3D.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkVector.h>
#include <vtkCamera.h>

#include <complex>
#include <vector>
#include <array>
// This is for complex numbers under CUDA:
#include <thrust/complex.h>

// #include <mpreal.h> 
#include <getopt.h>

// Helper macross
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

// Macro returning the linear index into matrix of
// dimensions Nc (cols), Nr (rows).  The linear index
// is row major since we are working in C.
#define LINDEX(Nr, Nc, r, c)  ((c) + (r)*(Nc))

// Display window dimensions
#define NX 700
#define NY 700
#define NY0 350

// Values used to distribute the jobs amongst the GPUs.
#define NT NX*NY
#define NTHD MIN(NT, 1024)
#define NBLK ((NT-1)/NTHD + 1)

// Default number of logistic map iterations.
#define NITER 8

// Amount to grow/shrink when turning mouse wheel.
#define SCALE 1.2

// Value at which to saturate poly (both pos and neg)
#define SAT 10.0

// VTK type declaration macro
#define MY_CREATE(type, name) \
    type *name = type::New()

// Global struct holding info about complex plane and iterations.
typedef struct {
  int N;          // Number of logistic map iterations (settable).
  double w, h;    // Width, height of image in real numbers.
  double xmin, xmax, ymin, ymax;  // bounds of plane
  double dx, dy;  // Step sizes in plane
  double xc, yc;  // Center point of plane.
  double *z;      // This is place to attach computed values of plane.
                  // I malloc the storage later, in main().
} ComplexPlane;
ComplexPlane Z;


//-----------------------------------------------------------------
// Declare fcns computing the Mandelbrot set in the complex plane.
void computeMandelbrot(vtkUniformGrid *imageData);
__global__
void f(double *z, double *lamr, double *lami, int N);

// Declare host-side graphics manipulation fcns.
void moveZoom(int i, int j, double zoom);
void moveTranslate(vtkVector<int, 4> p);
void MapIndexToPhysicalPoint(int i, int j, int k, double xyz[3]);
void insertZIntoImageData(vtkUniformGrid *imageData, double *z);


//--------------------------------------------------------
// Error checking wrapper around CUDA fcns.  Copied from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


//-------------------------------------------------------------
// Create most VTK objects as globals so I can access them from
// everywhere.  Some say it's bad practice to use globals, but
// I say it's easier than trying to pass around pointers to
// objects from my main prog to the callbacks in the custom
// interactor.  
MY_CREATE(vtkUniformGrid, rImageData);
//MY_CREATE(vtkUniformGrid, iImageData);
MY_CREATE(vtkImageMapToColors, colorComplexPlane);
MY_CREATE(vtkImageActor, imageActor);
MY_CREATE(vtkRenderer, renderer);
MY_CREATE(vtkCamera, camera);
MY_CREATE(vtkRenderWindow, renWin);
MY_CREATE(vtkRenderWindowInteractor, iren);


//------------------------------------------------------------------
// Custom mouse interactor -- started from
// https://kitware.github.io/vtk-examples/site/Cxx/Interaction/MouseEvents/
// and then modified for my purposes.
class customMouseInteractorStyle : public vtkInteractorStyleImage
{
public:
  static customMouseInteractorStyle* New();
  vtkTypeMacro(customMouseInteractorStyle, vtkInteractorStyleImage)

  vtkVector<int, 4> evt;    // Event coords -- down xy, up xy
  double scale;             // Scale to zoom in/out
  bool quit;
  
  void OnMouseWheelForward() override {
    std::cout << "MouseWheelForward ... ";
    int i = this->Interactor->GetEventPosition()[0];
    int j = this->Interactor->GetEventPosition()[1];
    std::cout << "[i,j] = [" << i << ", " << j << "]" << std:: endl;
    scale = 1.0/SCALE;
    moveZoom(i, j, scale);
    // Tell pipeline to update
    renderer->ResetCamera();
    renderer->GetActiveCamera( )->SetViewUp( 0,1,0 );
    renWin->Render();
  }

  void OnMouseWheelBackward() override {
    std::cout << "MouseWheelBackward ... ";
    // Note that the event position refers to the actual window size.
    // If the window was resized by the user, then i,j are different
    // from the size implied by the original window.
    int i = this->Interactor->GetEventPosition()[0];
    int j = this->Interactor->GetEventPosition()[1];
    std::cout << "[i,j] = [" << i << ", " << j << "]" << std:: endl;
    scale = SCALE;    
    moveZoom(i, j, scale);
    // Tell pipeline to update
    renderer->ResetCamera();
    renWin->Render();
  }
  
  void OnMiddleButtonDown() override
  {
    // This returns point in window where button went down.
    std::cout << " MiddleButtonDown ..." << std::endl;
    int i = this->Interactor->GetEventPosition()[0];
    int j = this->Interactor->GetEventPosition()[1];
    std::cout << "[i,j] = [" << i << ", " << j << "]" << std:: endl;
    evt[0] = i;
    evt[1] = j;
    vtkInteractorStyleImage::OnMiddleButtonDown();
    // Nothing to do here -- must wait until button pops up.
  }


  void OnMiddleButtonUp() override
  {
    std::cout << " MiddleButtonUp ..." << std::endl;
    int i = this->Interactor->GetEventPosition()[0];
    int j = this->Interactor->GetEventPosition()[1];
    std::cout << "[i,j] = [" << i << ", " << j << "]" << std:: endl;
    evt[2] = i;
    evt[3] = j;
    vtkInteractorStyleImage::OnMiddleButtonUp();
    moveTranslate(evt);
    // Tell pipeline to update
    renderer->ResetCamera();
    renWin->Render();
  }

  
  void OnLeftButtonDown() override {
    std::cout << "Left button down ..." << std::endl;
  }

  void OnLeftButtonUp() override {
    std::cout << "Left button up ..." << std::endl;
  } 

  void OnRightButtonDown() override {
    std::cout << "Right button down ..." << std::endl;
  }

  void OnRightButtonUp() override {
    std::cout << "Right button up ..." << std::endl;
  } 

  void OnKeyDown() override {
    std::cout << "Key down ..." << std::endl;
    std::string key = this->Interactor->GetKeySym();
    std::cout << "Key pressed: " << key << std::endl;
    if (key == "q"){
      quit = true;    
    } else {
      quit = false;    
    }
    this->Interactor->ExitCallback ();
  }

  vtkVector<int, 4> getEvt(void) {
    return evt;
  }

  double getScale(void) {
    return scale;
  }

};
vtkStandardNewMacro(customMouseInteractorStyle);
// Instantiate iStyle here, after defining it.
MY_CREATE(customMouseInteractorStyle, iStyle);

//===========================================================
// Manipulate the view of the complex plane.
void moveZoom(int i, int j, double zoom) {
  double xyz[3];
  int iz = 0;
  double x0, y0;
  double tx, ty;
  
  printf("Old xmin = %e, ymin = %e, xmax = %e, ymax = %e, w = %e, h = %e\n",
	 Z.xmin, Z.ymin, Z.xmax, Z.ymax, Z.w, Z.h);
  //printf("Old dx = %e, dy = %e\n", Z.dx, Z.dy);
  printf("Old xc = %e, yc = %e\n", Z.xc, Z.yc);  
  

  // Grab x,y coords of mouse point.
  MapIndexToPhysicalPoint (i, j, iz, xyz);  
  // The actual view image occupies the window from [0.16, 0.84].
  // I need to scale the zoom point using the approx 1.3 scaling.
  x0 = 1.3*(xyz[0]-0.5);  // New center of image in [0, 1] coords
  y0 = 1.3*(xyz[1]-0.5);  // New center of image in [0, 1] coords
  printf("Mouse zoom event at [x0,y0] = [%e, %e]\n", x0, y0);

  // Move center of image to [0,0], expand by zoom, then move back to
  // [x0,y0].  Assume that zoom < 1.
  tx = Z.xc+Z.w*x0;  // Translate image center to origin
  ty = Z.yc+Z.h*y0;
  printf("Translate ymin to %e, ymax to %e\n", Z.ymin - ty, Z.ymax - ty);
  Z.xmin = zoom*(Z.xmin - tx) + tx;  // Zoom and translate back.
  Z.ymin = zoom*(Z.ymin - ty) + ty;  
  Z.xmax = zoom*(Z.xmax - tx) + tx;
  Z.ymax = zoom*(Z.ymax - ty) + ty;  
  Z.xc = (Z.xmax+Z.xmin)/2.0;
  Z.yc = (Z.ymax+Z.ymin)/2.0;  
  Z.w = Z.xmax-Z.xmin;
  Z.h = Z.ymax-Z.ymin;
  Z.dx = (Z.xmax-Z.xmin)/(NX-1);
  Z.dy = (Z.ymax-Z.ymin)/(NY-1);
  printf("New xmin = %e, ymin = %e, xmax = %e, ymax = %e, w = %e, h = %e\n",
	 Z.xmin, Z.ymin, Z.xmax, Z.ymax, Z.w, Z.h);
  //printf("New dx = %e, dy = %e\n", Z.dx, Z.dy);
  printf("New xc = %e, yc = %e\n", Z.xc, Z.yc);
  
  // Now that I have an updated Z, must update ImageData
  rImageData->AllocateScalars(VTK_DOUBLE, 1);
  
  // Now compute Mandelbrot set using new origin and spacing.
  computeMandelbrot(rImageData);  // Compute the whole set.

  return;
}

//------------------------------------------------------
void moveTranslate(vtkVector<int, 4> p) {
  // Convert first pixel point to real number 
  int iz = 0;
  double xyz[3];
  double x1, y1, x2, y2;
  double myxmin, myymin, myh, myw;  // Locals used for checking only.
  double dalphax, dalphay;

  // Location of middle button down
  MapIndexToPhysicalPoint (p[0], p[1], iz, xyz);  
  x1 = xyz[0];
  y1 = xyz[1];
  printf("moveTranslate, x1 = %e, y1 = %e\n", x1, y1);

  // Location of middle button up
  MapIndexToPhysicalPoint (p[2], p[3], iz, xyz);  
  x2 = xyz[0];
  y2 = xyz[1];
  printf("moveTranslate, x2 = %e, y2 = %e\n", x2, y2);  

  printf("Old xmin = %e, ymin = %e, xmax = %e, ymax = %e, w = %e, h = %e\n",
	 Z.xmin, Z.ymin, Z.xmax, Z.ymax, Z.w, Z.h);
  //printf("Old dx = %e, dy = %e\n", Z.dx, Z.dy);
  printf("Old xc = %e, yc = %e\n", Z.xc, Z.yc);  
  
  // Amount to translate.
  dalphax = x2-x1;
  dalphay = y2-y1;

  // New min (origin) and max
  Z.xmin = Z.xmin - dalphax*Z.w;
  Z.xmax = Z.xmax - dalphax*Z.w;  
  Z.ymin = Z.ymin - dalphay*Z.h;
  Z.ymax = Z.ymax - dalphay*Z.h;  
  Z.xc = (Z.xmax+Z.xmin)/2.0;
  Z.yc = (Z.ymax+Z.ymin)/2.0;  
  Z.w = Z.xmax-Z.xmin;
  Z.h = Z.ymax-Z.ymin;
  Z.dx = (Z.xmax-Z.xmin)/(NX-1);
  Z.dy = (Z.ymax-Z.ymin)/(NY-1);

  printf("New xmin = %e, ymin = %e, xmax = %e, ymax = %e, w = %e, h = %e\n",
	 Z.xmin, Z.ymin, Z.xmax, Z.ymax, Z.w, Z.h);
  //printf("New dx = %e, dy = %e\n", Z.dx, Z.dy);
  printf("New xc = %e, yc = %e\n", Z.xc, Z.yc);  

  // For some reason I need to do this in order to refresh drawing.
  rImageData->AllocateScalars(VTK_DOUBLE, 1);
  
  // Now compute Mandelbrot set using new origin
  computeMandelbrot(rImageData);  // Compute the whole set.
  
  return;
}


void MapIndexToPhysicalPoint(int i, int j, int k, double xyz[3])
{
  // This replaces vtkImageData::TransformIndexToPhysicalPoint
  // Z is the complex plane global.  NX, NY are size of view plane in pixels.
  // The return elements are placed in xyz, the elements are doubles
  // between 0 and 1.
  double deltax, deltay;
  double alphax, alphay;
  double x, y, z;
  
  alphax = static_cast<double>(i)/static_cast<double>(NX);
  alphay = static_cast<double>(j)/static_cast<double>(NY);  
  z = 0.0;
  
  xyz[0] = alphax; // x;
  xyz[1] = alphay; // y;
  xyz[2] = z;
}



//======================================================================
//======================================================================
//======================================================================
int main(int argc, char* argv[])
{
  vtkNew<vtkNamedColors> colors;
  double x0, y0;
  int c;

  // This is initial view window
  x0 = -0.5;
  y0 = 0.0;
  Z.w = 3.0;
  Z.h = 3.0;
  Z.N = NITER;
  // Process command line args (if any)
  static struct option long_options[] =
    {
     {"x",  required_argument, 0, 'x'},
     {"y",  required_argument, 0, 'y'},
     {"w",  required_argument, 0, 'w'},
     {"h",  required_argument, 0, 'h'},
     {"N",  required_argument, 0, 'N'},       
     {0, 0, 0, 0}
    };
  /* getopt_long stores the option index here. */
  int option_index = 0;
  while (1) {
    c = getopt_long (argc, argv, "x:y:w:h:N:", long_options, &option_index);
    //std::cout << "c = " << c << std::endl;
    if (c == -1) break;
    switch (c) {
    case 'x':
      x0 = atof(optarg);
      //std::cout << "x = " << x0 << std::endl;
      break;
    case 'y':
      y0 = atof(optarg);
      //std::cout << "y = " << y0 << std::endl;      
      break;
    case 'w':
      Z.w = atof(optarg);
      //std::cout << "w = " << w << std::endl;      
      break;
    case 'h':
      Z.h = atof(optarg);
      //std::cout << "h = " << h << std::endl;            
      break;
    case 'N':
      Z.N = atoi(optarg);
      //std::cout << "N = " << N << std::endl;            
      break;
    case '?':
      fprintf (stderr,
               "Unknown option character 0x%x'.\n",
               optopt);
      return 1;
    default:
      abort ();
    }
  }
  printf("Starting x0 = %e, y0 = %e, w = %e, h = %e, N = %d\n", x0, y0, Z.w, Z.h, Z.N);  

  //---------------------------------------------------------
  // Finalize initialization of ComplexPlane Z -- create space to
  // hold the complex plane itself.
  // Malloc place to copy result back to host
  Z.z = (double *)malloc(NX*NY*sizeof(double));


  
  //--------------------------------------------------------------------

  // Map the scalar values in the image to colors with a lookup table
  // Play with these settings to alter the color map.
  vtkSmartPointer<vtkLookupTable> lookupTable =
    vtkSmartPointer<vtkLookupTable>::New();
  lookupTable->SetNumberOfTableValues(512);
  // I use sqrt just to get interesting colors
  lookupTable->SetTableRange(0, sqrt(Z.N-1));  
  //lookupTable->SetTableRange(0, log(Z.N-1));  
  lookupTable->SetAboveRangeColor(0.0, 0.0, 0.0, 1.0);
  lookupTable->SetNanColor(0.0, 0.0, 0.0, 1.0);
  //lookupTable->SetRampToLinear();
  lookupTable->SetRampToSQRT();
  //lookupTable->SetRampToSCurve();
  //lookupTable->SetScaleToLog10();
  lookupTable->SetScaleToLinear();
  //lookupTable->SetScaleToSQRT();  
  lookupTable->Build();

  //----------------------------------------------------------------
  // Colorbar to show off color map
  vtkSmartPointer<vtkScalarBarActor> scalarBar =
    vtkSmartPointer<vtkScalarBarActor>::New();
  scalarBar->SetLookupTable( lookupTable );
  scalarBar->SetOrientationToVertical();
  scalarBar->GetLabelTextProperty()->SetColor(0,0,1);
  scalarBar->GetTitleTextProperty()->SetColor(0,0,1);
  scalarBar->SetMaximumNumberOfColors(512);


  
  // Position scalarBar in window
  scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  scalarBar->SetPosition(0.85, 0.1);
  scalarBar->SetWidth(.10);
  scalarBar->SetHeight(0.8);


  //--------------------------------------------------------
  // Pass the original image and the lookup table to a
  // filter to create a color image.
  cout << "Configure colorComplexPlane ... " << endl;
  colorComplexPlane->SetLookupTable(lookupTable);
  colorComplexPlane->PassAlphaToOutputOn();
  colorComplexPlane->SetInputData(rImageData);  // set to real or imag plane

  // Configure initial ImageData
  cout << "Configure colorComplexPlane ... " << endl;
  Z.xmin = x0 - Z.w/2.0;
  Z.xmax = x0 + Z.w/2.0;  
  Z.ymin = y0 - Z.h/2.0;
  Z.ymax = y0 + Z.h/2.0;  
  Z.xc = (Z.xmax+Z.xmin)/2.0;
  Z.yc = (Z.ymax+Z.ymin)/2.0;  
  Z.dx = (Z.xmax-Z.xmin)/(NX-1);
  Z.dy = (Z.ymax-Z.ymin)/(NY-1);
  printf("xmin = %e, xmax = %e, ymin = %e, ymax = %e, dx = %e, dy = %e\n",
         Z.xmin, Z.xmax, Z.ymin, Z.ymax, Z.dx, Z.dy);

  // rImageData is the data  plane for display.  It is the unit
  // square
  rImageData->SetExtent( 0, NX-1, 0, NY-1, 0, 0 );  // Set image size in pixels
  rImageData->AllocateScalars(VTK_DOUBLE, 1); 
  
  // Compute initial Mandelbrot for display
  cout << "Compute initial Mandelbrot ... " << endl;  
  computeMandelbrot(rImageData);  // Compute the whole set.
  
  // Configure image actor.  Actor has built-in mapper.
  cout << "Configure image actor ... " << endl;    
  imageActor->InterpolateOff();
  imageActor->GetMapper()->SetInputConnection(colorComplexPlane->GetOutputPort());
  
  // Configure renderer
  cout << "Configure renderer ..." << endl;
  renderer->AddActor(imageActor);
  renderer->AddActor(scalarBar);
  renderer->SetBackground(colors->GetColor3d("MidnightBlue").GetData());
  camera->SetViewUp(0,1,0);
  //camera->SetFocalPoint(0.0, 1.0, 0.0);
  //camera->SetPosition(0,0,1);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  
  // Configure render window
  cout << "Configure render window ..." << endl;
  renWin->AddRenderer(renderer);
  renWin->SetSize(NX, NY); // set window size in pixels
  renWin->SetWindowName("Mandelbrot set in complex plane");
            
  // Configure interactor and interactor style
  iren->SetRenderWindow(renWin);
  iStyle->SetInteractor(iren);
  iren->SetInteractorStyle(iStyle);
  
  // Start rendering thread
  cout << "Start rendering thread......" << endl;  
  renWin->Render();

  cout << "Initialize interactor......" << endl;  
  iren->Initialize();
  
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Start interactor event loop......" << std::endl;
  iren->Start();

  // If I get here, it's because the event loop terminated.
  if (iStyle->quit == true) {
    std::cout << "User requested quit.  Exiting ..." << std::endl;
    return 0;
  } else {
    std::cout << "Returned from event loop for unknown reasons." << std::endl;
    return -1;
  }

}


//====================================================================
__host__
void insertZIntoImageData(vtkUniformGrid *imageData, double *z) {
  int ix, iy, iz;
  double *pixel;

  // Insert returned z values into imageData
  iz = 0;
  for (ix = 0; ix < NX; ix++) {
    for (iy = 0; iy < NY; iy++) {
      pixel = static_cast<double*>(imageData->GetScalarPointer(ix, iy, iz));
      // Take sqrt to get interesting colors.  No other reason.
      *pixel = sqrt(z[LINDEX(NY, NX, iy, ix)]);
      //printf("z[%d,%d] = %f\n", ix, iy, *pixel);
    }
  }
}



//=====================================================================
__host__
void computeMandelbrot(vtkUniformGrid *imageData) {
  // This computes the Mandelbrot set using the current values of the
  // complex plane Z.  It then sticks the updated set into
  // imageData so it can be displayed.
  int ix, iy;
  double lamr[NX*NY];
  double lami[NX*NY];
  
  std::cout << "Use GPUs to compute Mandelbrot set...." << std::endl;
  
  //-------------------------------------------------------------------
  // Now set up CUDA stuff
  

  //First attempt at assigning two GPUs
  cudaSetDevice(0)
  //I want half the NY so I wil create a new variable called NY0 which is NY/2
  //int NY0 = NY/2 (I will define it at the start cuz this did not work)

  // The value of lambda at each point in complex plane
  double *dlamr;
  gpuErrchk( cudaMalloc((void**)&dlamr, NX*NY*sizeof(double)) );  
  double *dlami;
  gpuErrchk( cudaMalloc((void**)&dlami, NX*NY*sizeof(double)) );
  
  // Make local lambda plane
  for (ix = 0; ix < NX; ix++) {
    for (iy = 0; iy < NY; iy++) {
      lamr[LINDEX(NY, NX, iy, ix)] = Z.xmin + ix*Z.dx;
      lami[LINDEX(NY, NX, iy, ix)] = Z.ymin + iy*Z.dy;
    }
  }
  // Copy lambda plane values to device
  gpuErrchk( cudaMemcpy(dlamr, lamr, NX*NY*sizeof(double),
                        cudaMemcpyHostToDevice));
  gpuErrchk( cudaMemcpy(dlami, lami, NX*NY*sizeof(double),
                        cudaMemcpyHostToDevice));

  
  // Malloc complex plane used to iterate the fcn on the device
  // No need to copy anything here -- the plane's values will
  // be generated on the device.
  double *dz;
  gpuErrchk( cudaMalloc((void**)&dz, NX*NY*sizeof(double)) );

  // Call fcn running on GPUs to iterate map N times.
  //printf("Calling f, [NBLK, NTHD] = [%d, %d], N = %d\n", NBLK, NTHD, Z.N);
  f<<<NBLK,NTHD>>>(dz, dlamr, dlami, Z.N);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  // Copy dz back to host after iteration.
  gpuErrchk( cudaMemcpy(Z.z, &(dz[0]), NX*NY*sizeof(double),
                        cudaMemcpyDeviceToHost) );

  // Insert returned z values into imageData
  insertZIntoImageData(imageData, Z.z);

  //To free up the memory in the GPU device
  gpuErrchk(cudaFree(dlamr));
  gpuErrchk(cudaFree(dlami));
  gpuErrchk(cudaFree(dz));

  std::cout << " ... done!\n" << std::endl;
  return;
}

//---------------------------------------------------------------
// This fcn iterates a point in the complex plane.
__global__
void f(double *z, double *lamr, double *lami, int N) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ltid = threadIdx.x;   // my local index on this block.
  int bid = blockIdx.x;
  int i,j;
  int k;
  
  // Figure out which lambda value to use based on my
  // block and thread index values.
  i = (int) tid/NX;
  j = (int) tid%NX;
  //printf("Entered f, N = %d, [i,j] = [%d, %d]\n", N, i, j);
  thrust::complex<double> mylam(lamr[LINDEX(NY, NX, j, i)],
				lami[LINDEX(NY, NX, j, i)]);
  //printf("mylam = [%f, %f]\n", mylam.real(), mylam.imag());

  // Modify this to choose between Mandelbrot and Logistic iteration.
  //thrust::complex<double> x(0.5, 0.0);   // Logistic
  thrust::complex<double> x(0.0, 0.0);  // Mandelbrot

  // Do iteration.  If x escapes, then  break.
  for (k=0; k<N; k++) {
    //x = mylam*x*(1.0-x);  // Logistic
    x = x*x + mylam;    // Mandelbrot
    //if ((x.real()*x.real() + x.imag()*x.imag()) > 4.0) {
    if ((x.real()*x.real() + x.imag()*x.imag()) > 2.0) {
      break;
    }
  }

  // Put count to escape into z.
  z[LINDEX(NY, NX, j, i)] = (double) k;

}


