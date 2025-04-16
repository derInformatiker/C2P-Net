import json
from random import sample
import vtk
import os
import vtkutils


def save_poly(points, output_path):

    poly = vtk.vtkPolyData()

    poly.SetPoints(points)

    cellsArray = vtk.vtkCellArray()

    for p in range(points.GetNumberOfPoints()):
        cellsArray.InsertNextCell( 1, (p, ) )
    poly.SetPolys(cellsArray)

    print("poly written to", output_path)

    vtkutils.write_mesh(poly,
    output_path)

    return poly

def load_landmarks(marker_path):
    file_path = marker_path

    with open(file_path, "r") as f:
        marker = json.load(f)
    print(len(marker["markups"][0]["controlPoints"]))

    # print(marker["markups"][0]["controlPoints"])

    points = vtk.vtkPoints()

    for p in marker["markups"][0]["controlPoints"]:
        points.InsertNextPoint(p["position"])
    print(points.GetNumberOfPoints())
    return points

def convert_markers_json_to_ply():
    d2ear_folder = "/home/liupeng/2_Data/D2EAR_data/07_26/D2EAR"

    marker_folders = list(filter(lambda x:str(x).startswith("2020") and os.path.isdir(os.path.join(d2ear_folder, x)), os.listdir(d2ear_folder)))

    # marker_path = "/home/liupeng/2_Data/D2EAR_data/07_26/D2EAR/SSM neu/MarkupsCurve_2.mrk.json"

    marker_folders = [os.path.join(d2ear_folder, x) for x in marker_folders]

    for marker_folder in marker_folders:

        print("================ entering {} ...=====================".format(marker_folder))

        marker_files = list(filter(lambda x:str(x).endswith("json"), os.listdir(marker_folder)))

        print(marker_files)

        for file_name in marker_files:

            points = load_landmarks(os.path.join(marker_folder, file_name))

            file_name_output = file_name[:-8] + "ply"

            poly = save_poly(points, os.path.join(marker_folder, file_name_output))


        poly = vtkutils.load_mesh(os.path.join(marker_folder, file_name_output))

        print(poly.GetNumberOfPoints())


def convert_markers_json_to_ply_per_sample(sample_folder, marker_json_list=None, output_folder_name=None):

    if output_folder_name:
        output_folder = os.path.join(sample_folder, output_folder_name)

    for marker_json_file in marker_json_list:
        points = load_landmarks(os.path.join(sample_folder, marker_json_file))

        file_name_output = marker_json_file[:-8] + "ply"
        print(output_folder)
        poly = save_poly(points, os.path.join(output_folder, file_name_output))

        poly = vtkutils.load_mesh(os.path.join(output_folder, file_name_output))

        print(poly.GetNumberOfPoints())


if __name__ == "__main__":

# convert_markers_json_to_ply()

    sample_folder = "mesh_dataset/ear_data_real/2020-009_right/"

    marker_list = [

    "long process of incus.mrk.json",

    "lateral Process.mrk.json",

    "Anulus.mrk.json", 

    "malleus handle.mrk.json", 

    "Umbo.mrk.json"

    ]


    convert_markers_json_to_ply_per_sample(sample_folder=sample_folder,
    marker_json_list=marker_list,
    output_folder_name="landmarks")