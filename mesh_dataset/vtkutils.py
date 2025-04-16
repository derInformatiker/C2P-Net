# Written by Micha Pfeiffer vom NCT Dresden
# from pcl import load
from vtk import (vtkSphereSource, vtkPolyData, vtkDecimatePro)
import vtk, os
from vtk.util import numpy_support
import random
from vtk import *
import math

def load_mesh(filename):

    """
    Loads a mesh using VTK. Supported file types: stl, ply, obj, vtk, vtu, vtp, pcd.

    Arguments:
    ---------
    filename (str)

    Returns:
    --------
    vtkDataSet
                which is a vtkUnstructuredGrid or vtkPolyData, depending on the file type of the mesh.
    """
    # Load the input mesh:
    fileType = filename[-4:].lower()
    if fileType == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtk":
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vts":
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    # elif fileType == ".pcd":
    #     import pcl
    #     pc = pcl.load(filename)
    #     pts = vtk.vtkPoints()
    #     verts = vtk.vtkCellArray()
    #     for i in range(pc.size):
    #             pts.InsertNextPoint(pc[i][0], pc[i][1], pc[i][2])
    #             verts.InsertNextCell(1, (i,))
    #     mesh = vtkPolyData()
    #     mesh.SetPoints(pts)
    #     mesh.SetVerts(verts)

    else:
        raise IOError(
                "Mesh should be .vtk, .vtu, .vtp, .obj, .stl, .ply or .pcd file!")

    if mesh.GetNumberOfPoints() == 0:
        raise IOError("Could not load a valid mesh from {}".format(filename))
    return mesh


def write_mesh(mesh, filename):
    """
    Saves a VTK mesh to file. 
    Supported file types: stl, ply, obj, vtk, vtu, vtp, pcd.

    Arguments:
    ---------
    mesh (vtkDataSet):
            mesh to save
    filename (str): 
            name of the file where to save the input mesh. MUST contain the desired extension.

    """

    if mesh.GetNumberOfPoints() == 0:
        raise IOError("Input mesh has no points!")

    # Get file format
    fileType = filename[-4:].lower()
    if fileType == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".obj":
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".ply":
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtk":
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtu":
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vts":
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    else:
        raise IOError(
            "Supported extensions are .vtk, .vtu, .vts, .vtp, .obj, .stl, .ply!")


def decimate_mesh(input_mesh, reduction, v=False):

    # sphereS = vtkSphereSource()
    # sphereS.Update()

    # inputPoly = vtkPolyData()
    # inputPoly.ShallowCopy(sphereS.GetOutput())
    if v:
        print("Before decimation\n"
            "-----------------\n"
            "There are " + str(input_mesh.GetNumberOfPoints()) + " points.\n"
            "There are " + str(input_mesh.GetNumberOfPolys()) + " polygons.\n")

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(input_mesh)
    decimate.SetTargetReduction(reduction) # .50
    decimate.Update()

    decimated = decimate.GetOutput()
    # decimatedPoly = vtkPolyData()
    # decimatedPoly.ShallowCopy(decimate.GetOutput())
    if v:
        print("After decimation \n"
            "-----------------\n"
            "There are " + str(decimated.GetNumberOfPoints()) + " points.\n"
            "There are " + str(decimated.GetNumberOfPolys()) + " polygons.\n")
    return decimated


def createPolyData( verts, tris):
    """Create and return a vtkPolyData.
    
    verts is a (N, 3) numpy array of float vertices

    tris is a (N, 1) numpy array of int64 representing the triangles
    (cells) we create from the verts above.  The array contains 
    groups of 4 integers of the form: 3 A B C
    Where 3 is the number of points in the cell and A B C are indexes
    into the verts array.
    """
    
    # save, we share memory with the numpy arrays
    # so they can't get deleted
    
    poly = vtk.vtkPolyData()
    
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(verts))        
    poly.SetPoints(points)
    
    # cells = vtk.vtkCellArray()
    # cells.SetCells(len(tris) / 4, numpy_support.numpy_to_vtkIdTypeArray(tris))        
    # poly.SetPolys(cells)
    
    return poly



def storeTransformationMatrix( grid, tf ):
    mat = tf.GetMatrix()
    matArray = vtk.vtkFloatArray()
    matArray.SetNumberOfTuples(16)
    matArray.SetNumberOfComponents(1)
    matArray.SetName( "TransformationMatrix" )
    for row in range(0,4):
        for col in range(0,4):
            matArray.SetTuple1( row*4+col, mat.GetElement( row, col ) )
    grid.GetFieldData().AddArray(matArray)
    return True


def scale_model(model, scale_matrix=(1,1,1), store_transform=False):

    transform = vtk.vtkTransform()
    transform.Scale(scale_matrix)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(model)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    model_scaled = transformFilter.GetOutput()
    print("model scaled by matrix:", scale_matrix)
    if store_transform:
        storeTransformationMatrix(grid=model, tf=transform)
    return model_scaled


def center_mesh(mesh):
    tf = vtk.vtkTransform()
    bounds = [0]*6;
    mesh.GetBounds(bounds)
    dx = -(bounds[1]+bounds[0])*0.5
    dy = -(bounds[3]+bounds[2])*0.5
    dz = -(bounds[5]+bounds[4])*0.5
    print("Moving point cloud by:", (dx,dy,dz) )
    tf.Translate( (dx,dy,dz) )

    tfFilter = vtk.vtkTransformFilter()
    tfFilter.SetTransform( tf )
    tfFilter.SetInputData( mesh )
    tfFilter.Update()
    mesh = tfFilter.GetOutput()
    return mesh


def extract_surface(inputMesh):
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(inputMesh)
    surfaceFilter.Update()
    surface = surfaceFilter.GetOutput()

    return surface


def combine_surface(mesh_list):
    append_filter = vtk.vtkAppendFilter()
    for s in range(0, len(mesh_list)):
        append_filter.AddInputData(mesh_list[s])
        append_filter.Update()
    combined = append_filter.GetOutput()
    return combined


def combine_poly(poly_list):
    append_filter = vtk.vtkAppendPolyData()
    for p in range(0, len(poly_list)):
        append_filter.AddInputData(poly_list[p])
        append_filter.Update()
    combined = append_filter.GetOutput()
    return combined


def unstructuredGridToPolyData(ug):
    """
    Converts an input unstructured grid into a polydata object. 
    Be careful since PolyData objects cannot contain 3D elements, thus all 
    tetrahedra will be lost with this operation.

    Parameters
    ----------
    ug (vtkUnstructuredGrid):
            The input unstructured grid

    Returns
    ----------
    vtkPolyData

    """
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(ug)
    geometryFilter.Update()
    return geometryFilter.GetOutput()


def polyDataToUnstructuredGrid(pd):
    """
    Converts an input polydata into an unstructured grid.

    Parameters
    ----------
    pd (vtkPolyData):
            The input polydata

    Returns
    ----------
    vtkUnstructuredGrid

    """
    appendFilter = vtkAppendFilter()
    appendFilter.SetInputData(pd)
    appendFilter.Update()
    return appendFilter.GetOutput()




def randomSurface( fullSurface, wDistance=1, wNormal=1, wNoise=1, surfaceAmount=None, centerPointID=None, hints_flag=False ):
    """
    Extract a random part of a surface mesh. Starting from a (random) center node c, the
    surrounding points p are assigned three values:
        - Geodesic distance from the center node c
        - Angle difference between the normal of p and the normal of c
        - Random perlin noise value, sampled at the positon p.
    From these three values, we build a weighted sum, which acts as the likelyhood of a point
    being removed. We then select a threshold and remove all points whose likelyhood exceeds
    this threshold. The remaining points are the selected surface.
    The weights in the weighted sum can be used to influence whether the geodesic distance,
    the normals or the random perlin noise value should have a higher influence on the
    removal.

    Arguments:
    ----------
    fullSurface (vtkPolyData):
        Full surface. A part of this mesh will be returned. Point IDs and Cell IDs will not
        be kept.
    wDistance (float):
        Influence of the geodesic distance of a point to the center point c.
        If this value is > 0 and the other weights are == 0, only the distance will be
        taken into account.
    wNormal (float):
        Influece of the angle between normals.
        If this value is > 0 and the other weights are == 0, only points with a similar
        normal as the center point will be selected.only points with a similar
        normal as the center point will be selected.
    wNoise (float):
        Influence of the highest noise.
        If this value is > 0 and the other weights are == 0, entirely random parts of the
        surface will be selected.
    surfaceAmount (float):
        Amount of the surface which we want to select. If None, a random amount
        between 0 and 1 will be selected.
        Valid range: (0,1)
    centerPointID (int):
        Index of the center point c. If None, a random index of all the surface indices will be selected.

    Returns:
    ----------
    vtkPolyData
        Describes the part of the fullSurface which was selected
    """

    # Decide how many points we want to select:
    if surfaceAmount is None:
        surfaceAmount = random.random()
    assert surfaceAmount <= 1 and surfaceAmount > 0, "surfaceAmount must be between 0 and 1."

    # if hints_flag:
    #     if surfaceAmount < 0.12:
    #         surfaceAmount = 0.12
    # print("surface amount:", surfaceAmount)
    pointsToSelect = surfaceAmount*fullSurface.GetNumberOfPoints()

    # if hints_flag:
    #     if pointsToSelect < 30:
    #         print("pointsToSelect too small, choose a greater one, now is:", pointsToSelect)
            # pointsToSelect = random.randint()


    
    fullSurface = generatePointNormals( fullSurface )
    normals = fullSurface.GetPointData().GetArray( "Normals" )

    # Store the IDs of all triangles:
    #triangleIDs = []
    #for i in range(0,fullSurface.GetNumberOfCells()):
    #    if fullSurface.GetCell(i).GetCellType() == VTK_TRIANGLE:
    #        triangleIDs.append(i)

    # Select a random point on the surface around which to center the selected part, if it is not provided:
    if centerPointID is None:
        centerPointID = random.randint(0,fullSurface.GetNumberOfPoints()-1)
    centerPoint = fullSurface.GetPoint( centerPointID )

    distance = geodesicDistance( fullSurface, centerPointID )
    fullSurface.GetPointData().AddArray( distance )


    # Get normal of that point:
    centerPointNormal = normals.GetTuple3( centerPointID )


    # Decrease with:
    # - distance from center point
    # - normal difference
    # - perlin noise

    noise = vtkPerlinNoise()
    noise.SetFrequency( 15, 15, 15 )
    noise.SetPhase( random.random()*150, random.random()*150, random.random()*150 )

    # Create an array which will be filled and then used for thresholding
    likelyhood = vtkDoubleArray()
    likelyhood.SetNumberOfComponents(1)
    likelyhood.SetNumberOfTuples(fullSurface.GetNumberOfPoints())
    likelyhood.SetName("likelyhood")

    minVal = 99999
    maxVal = -1
    for i in range( fullSurface.GetNumberOfPoints() ):
        pt = fullSurface.GetPoint( i )
        dist = math.sqrt( vtkMath.Distance2BetweenPoints(centerPoint, pt) )

        normal = normals.GetTuple3( i )
        dot = vtkMath.Dot( centerPointNormal, normal )
        normalAng = math.acos( max( min( dot, 1 ), -1 ) )

        rnd = abs(noise.EvaluateFunction( pt ))

        curLikelyhood = wDistance*dist + wNormal*normalAng + wNoise*rnd
        likelyhood.SetTuple1( i, curLikelyhood )
        minVal = min( minVal, curLikelyhood )
        maxVal = max( maxVal, curLikelyhood )

    #print("Likelyhood range:", minVal, maxVal)

    # Build histogramm of likelyhoods:
    histBins = 50
    hist = [0]*histBins
    for i in range( fullSurface.GetNumberOfPoints() ):
        l = likelyhood.GetTuple1( i )
        curBin = int(l/maxVal*(histBins-1))
        hist[curBin] += 1
   
    # Find out where to set the threshold so that surfaceAmount points are selected.
    # We do this by going through the histogram and summing up the values in the bins. As
    # soon as more than surfaceAmount points are selected, 
    thresholdedPoints = 0
    threshold = maxVal  # Start with default of selecting everything
    for i in range( histBins ):
        thresholdedPoints += hist[i]
        if thresholdedPoints >= pointsToSelect:
            threshold = (i+1)/histBins*maxVal
            break
    #print("Selected threshold", threshold)

    fullSurface.GetPointData().AddArray( likelyhood )

    likelyhoodRange = maxVal - minVal

    #rndThreshold = (random.random()*0.75 + 0.25)*likelyhoodRange + minVal

    thresh = vtkThreshold()
    thresh.SetInputData( fullSurface )
    thresh.SetInputArrayToProcess( 0,0,0,
            vtkDataObject.FIELD_ASSOCIATION_POINTS, "likelyhood" )
    thresh.ThresholdBetween( 0, threshold )
    thresh.Update()

    # Write resulting surface to file:
    # Debug output:
    # writer = vtkXMLPolyDataWriter()
    # writer.SetFileName( "partialSurfaceLikelyhood.vtp" )
    # writer.SetInputData( fullSurface )
    # writer.Update()

    partialSurface = unstructuredGridToPolyData( thresh.GetOutput() )

    fullArea = surfaceArea( fullSurface )
    partialArea = surfaceArea( partialSurface )

    #print( "Original area:", fullArea )
    #print( "Partial area:", partialArea )
    #print( "Selected amount: {:.2f}% (Target was {:.2f}%)".format( 100*partialArea/fullArea,
    #    100*surfaceAmount ) )

    return partialSurface


def generatePointNormals( surface ):
	# Check if the point normals already exist
	if surface.GetPointData().HasArray("Normals"):
		return surface

	# If no normals were found, generate them:
	normalGen = vtkPolyDataNormals()
	normalGen.SetInputData( surface )
	normalGen.ComputePointNormalsOn()
	normalGen.ComputeCellNormalsOff()
	normalGen.SplittingOff()        # Don't allow generator to add points at sharp edges
	normalGen.Update()
	return normalGen.GetOutput()

def geodesicDistance( surface, centerNodeID ):
 
	# pre-compute cell neightbors:
	neighbors = {}
	for i in range( surface.GetNumberOfPoints() ):
		neighbors[i] = getConnectedVertices( surface, i )
	
	distance = vtkDoubleArray() 
	distance.SetNumberOfTuples( surface.GetNumberOfPoints() )
	distance.SetNumberOfComponents( 1 )
	distance.Fill( 1e10 )   # initialize with large numbers
	distance.SetName( "geodesic_distance" )

	front = [centerNodeID]
	distance.SetTuple1( centerNodeID, 0 )

	while len(front) > 0:

		curID = front.pop(0)
		curPt = surface.GetPoint( curID )
		curDist = distance.GetTuple1( curID )
		curNeighbors = neighbors[curID]

		# Go through all neighboring points. Check if the distance in those points
		# is still up to data or whether there is a shorter path to them:
		for nID in curNeighbors:

			# Find distance between this neighbour and the current point:
			nPt = surface.GetPoint( nID )
			dist = math.sqrt( vtkMath.Distance2BetweenPoints( nPt, curPt ) )

			newDist = dist + curDist
			if newDist < distance.GetTuple1( nID ):
				distance.SetTuple1( nID, newDist )
				if not nID in front:
					front.append( nID )     # This neighbor node needs to be checked again!
		
	return distance

def getClosestPoints( mesh1, mesh2, subset=None, discardDuplicate=False ):
	"""
	For each point in mesh1, returns the index of the closest point in mesh2. 
	
	Parameters
	----------
	mesh1 (vtkDataSet):
		Topology of the first mesh
	mesh2 (vtkDataSet):
		Topology of the second mesh
	subset (list of ints):
		If specified, a subset of mesh1 nodes are considered. Subset represents
		the list of node indices to consider.
		Default: None
	discardDuplicate (bool):
		If true, the returned indices cannot be present more than once.
		Default: False

	Returns
	----------
	list of int:
		IDs of mesh2 vertices closest to each mesh1 vertex.

	"""
	locator = vtkPointLocator( )
	locator.SetDataSet( mesh2 )
	locator.SetNumberOfPointsPerBucket(1)
	locator.BuildLocator()

	mesh2IDs = []
	if subset is None:
		subset = range(mesh1.GetNumberOfPoints())
		
	for idx in subset:
		mesh1Point = mesh1.GetPoint(idx)
		mesh2PointID = locator.FindClosestPoint( mesh1Point )
		# If we want to discard duplicate indices and we have already found it, skip append
		if discardDuplicate and (mesh2PointID in mesh2IDs):
			pass
		else:
			mesh2IDs.append( mesh2PointID )
	return mesh2IDs
def getConnectedVertices( mesh, nodeID ):
	"""
	Find the neighbor vertices of the node with ID nodeID.

	Parameters
	----------
	mesh (vtkDataSet):
		The mesh topology
	nodeID (int):
		The ID of the node for which neighbors are to be found

	Returns
	----------
	list of int:
		IDs of the vertices that are neighbors of nodeID.

	"""
	connectedVertices = []

	#get all cells that vertex 'id' is a part of
	cellIDList = vtkIdList()
	mesh.GetPointCells( nodeID, cellIDList )
	
	for i in range(cellIDList.GetNumberOfIds()):
		c = mesh.GetCell( cellIDList.GetId(i) )
		pointIDList = vtkIdList()
		mesh.GetCellPoints( cellIDList.GetId(i), pointIDList )
		for j in range(pointIDList.GetNumberOfIds()):
			neighborID = pointIDList.GetId(j)
			if neighborID != nodeID:
				if not neighborID in connectedVertices:
					connectedVertices.append( neighborID )
	return connectedVertices

def surfaceArea( mesh ):
	area = 0
	for i in range(mesh.GetNumberOfCells()):
		if mesh.GetCell(i).GetCellType() == VTK_TRIANGLE:
			p0 = mesh.GetCell(i).GetPoints().GetPoint(0)
			p1 = mesh.GetCell(i).GetPoints().GetPoint(1)
			p2 = mesh.GetCell(i).GetPoints().GetPoint(2)
			a = mesh.GetCell(i).TriangleArea(p0,p1,p2)
			area += a

	areaArr = makeSingleFloatArray( "surfaceArea", area )
	mesh.GetFieldData().AddArray(areaArr)

	return area
def makeSingleFloatArray( name, val ):
	arr = vtkFloatArray()
	arr.SetNumberOfTuples(1)
	arr.SetNumberOfComponents(1)
	arr.SetTuple1(0,val)
	arr.SetName(name)
	return arr

def displ_field_interpolate(mesh, field, output_dir, tf_flag=False, output_filename=None):
    scale = 1 # default
    try:
        tf = loadTransformationMatrix( field )
        tf.Inverse()
        # print("Applying transform")
        tfFilter = vtkTransformFilter()
        tfFilter.SetTransform( tf )
        tfFilter.SetInputData( field )
        tfFilter.Update()
        field = tfFilter.GetOutput()
        
        # Apply transformation also to all vector fields:
        applyTransformation( field, tf )
        
        scale = tf.GetMatrix().GetElement(0,0)

    except Exception as e:
        print(e)
        print("Could not find or apply transformation. Skipping.")
    
    # writer = vtkXMLStructuredGridWriter()
    # writer.SetInputData( field )
    # writer.SetFileName( os.path.join( output_dir, "field.vts" ) )
    # writer.Update()
    # print("Written1")

    # Threshold to ignore all points outside of field:
    threshold = vtkThreshold()
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "preoperativeSurface")
    threshold.ThresholdByLower(0)
    threshold.SetInputData( field )
    threshold.Update()
    fieldInternal = threshold.GetOutput()

    # print("Scale", scale)

    if mesh.GetPointData().HasArray("estimatedDisplacement"):
        # print("found former estimatedDisplacement field, removing...")
        mesh.GetPointData().RemoveArray("estimatedDisplacement")

    kernel = vtkGaussianKernel()
    kernel.SetRadius(0.01*scale) 
    kernel.SetKernelFootprintToRadius()
    #kernel.SetKernelFootprintToNClosest()
    #kernel.SetNumberOfPoints( 4 )

    interpolator = vtkPointInterpolator()
    interpolator.SetKernel( kernel )
    interpolator.SetNullPointsStrategyToMaskPoints()
    interpolator.SetValidPointsMaskArrayName( "validInternalPoints" )
    #interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.SetSourceData( fieldInternal )
    interpolator.SetInputData( mesh )
    interpolator.Update()
    output = interpolator.GetOutput()

    # writer = vtkXMLUnstructuredGridWriter()
    # writer.SetInputData( fieldInternal )
    # writer.SetFileName( os.path.join( output_dir, "fieldInternal.vtu" ) )
    # writer.Update()
    # print(333)

    append = vtkAppendFilter()
    append.AddInputData( output )
    append.Update()
    output = append.GetOutput()

    # if save == True:
    # if not output_filename:
    #     output_filename = "initSurface_hints_withDispl.vtu"
    # else:
    #     output_filename = output_filename
    if output_filename:
        print("write mesh:", os.path.join( output_dir, output_filename ))
        writeMesh(output, os.path.join( output_dir, output_filename ))
    # writer = vtkXMLUnstructuredGridWriter()
    # writer.SetInputData( output )

    # writer.SetFileName( os.path.join( output_dir, output_filename ) )
    # writer.Update()

    return output
    
    
def warp_mesh(mesh_with_displ, displacement_field_name, output_folder = None, output_filename = None):
    if not mesh_with_displ.GetPointData().HasArray(displacement_field_name):
        print("mesh don't have the array {}, please check!".format(displacement_field_name))
        return None
    # print("warping by vector {}".format(displacement_field_name))
    # hints_array_
    mesh_with_displ.GetPointData().SetActiveVectors(displacement_field_name)
    warpVector = vtkWarpVector()
    warpVector.SetInputData(mesh_with_displ)
    warpVector.Update()
    output_warped = warpVector.GetOutput()
    # polydata.SetActiveVectors(warpData.GetName())
    # print("before removing double faces:", output_warped.GetNumberOfCells())
    # output_warped = removeDoubleFaces(output_warped)
    # print("after removing double faces:", output_warped.GetNumberOfCells())
    # print("chec"(output_warped.GetNumberOfPoints() * 2 - 4) == output_warped.GetNumberOfCells())

    if output_filename:
        print("writing warped mesh to {}".format(os.path.join( output_folder, output_filename)))
        # writer = vtkXMLUnstructuredGridWriter()
        # writer.SetInputData( output_warped )
        # writer.SetFileName( os.path.join( output_folder, output_filename) )
        # writer.Update()
        writeMesh(output_warped, os.path.join( output_folder, output_filename))


    return output_warped



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Test vtk utils.")
    parser.add_argument("model_path", type=str, help="Model path")  
    parser.add_argument("--output_path", type=str, help="Output path")

    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output_path

    # model = load_mesh(model_path)
    # print(model.GetNumberOfPoints())
    # # print(model.GetPoint(1000))
    # model = scale_model(model, scale_matrix=(2, 2, 2))
    # write_mesh(model, model_path)
    # print(model.GetPoint(1000))

    # output_path = args.output_path
    # write_mesh(model, output_path)


    # stl_files = os.listdir(model_path)
    # stl_files = list(filter(lambda x:x.endswith(".stl"), stl_files))
    # print(stl_files)

    # model_list = []
    # for f in stl_files:
    #     if not "Wall" in f:
    #         print(f)
    #         model_list.append(load_mesh(os.path.join(model_path, f)))
    
    # print("combining meshes...")
    # mesh_combined = combine_surface(model_list)
    # print(mesh_combined.GetNumberOfPoints())
    # print("scaling meshes")
    # mesh_combined = unstructuredGridToPolyData(mesh_combined)
    # print(mesh_combined.GetNumberOfPoints())
    # mesh_combined = scale_model(mesh_combined, scale_matrix=(0.0001, 0.0001, 0.0001))
    # print(mesh_combined.GetNumberOfPoints())
    # write_mesh(mesh_combined, os.path.join(args.output_path, "combined_without_wall.vtp"))
    # combined_model = load_mesh(model_path)
    # print(combined_model.GetNumberOfPoints())
    # combined_model = decimate_mesh(combined_model)
    # print(combined_model.GetNumberOfPoints())

    # write_mesh(combined_model, os.path.join(output_path, "combined_without_wall_low_poly.vtp"))


    # model = load_mesh(model_path)
    # model = center_mesh(mesh=model)
    # write_mesh(model, os.path.join(model_path))

    landmarks_path = "/home/liupeng/2_Data/D2EAR_data/07_26/D2EAR/SSM neu/MarkupsCurve_2.ply"
    # ssm_model =


    landmarks = load_mesh()
    # get_closest_point(None, None)