import slicer
import glob

for nrrd_path in glob.glob(f'D:/mesh2mesh/mesh_dataset/DIOME_FanShapeCorr/sample_*/annotations/seg_0.seg.nrrd'):
    # swap nrrd to stl
    segmentation = slicer.util.loadSegmentation(nrrd_path)
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(nrrd_path[:-26], segmentation)
    print('Finished', nrrd_path)