/*
Copyright (C) 2017  Liangliang Nan
https://3d.bk.tudelft.nl/liangliang/ - liangliang.nan@gmail.com

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include "../basic/logger.h"
#include "../model/point_set.h"
#include "../model/map.h"
#include "../method/method_global.h"
#include "../method/hypothesis_generator.h"
#include "../method/face_selection.h"
#include "../model/map_io.h"
#include "../model/point_set_io.h"

int main(int argc, char **argv)
{
    // initialize the logger (this is not optional)
    Logger::initialize();

    // below are the default parameters (change these when necessary)
    if (argc != 7)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <subsampling_factor> <lambda_data_fitting> <lambda_model_coverage> <lambda_model_complexity>" << std::endl;
        return EXIT_FAILURE;
    }

    // input point cloud file name
    const std::string input_file = argv[1];
    // output mesh file name
    const std::string output_file = argv[2];
    // subsampling factor
    const double subsampling_factor = std::stod(argv[3]);
    // lambda_data_fitting value
    const double lambda_data_fitting = std::stod(argv[4]);
    // lambda_model_coverage value
    const double lambda_model_coverage = std::stod(argv[5]);
    // lambda_model_complexity value
    const double lambda_model_complexity = std::stod(argv[6]);

    Method::lambda_data_fitting = lambda_data_fitting;
    Method::lambda_model_coverage = lambda_model_coverage;
    Method::lambda_model_complexity = lambda_model_complexity;

    // load point cloud from file
    PointSet *pset = PointSetIO::read(input_file);
    if (!pset)
    {
        std::cerr << "failed loading point cloud from file: " << input_file << std::endl;
        return EXIT_FAILURE;
    }

    // subsample the point cloud
    std::cout << "subsampling point cloud..." << std::endl;
    std::vector<unsigned int> sampled_indices;
    for (unsigned int i = 0; i < pset->num_points(); i++)
    {
        if (rand() / (RAND_MAX + 1.0) < subsampling_factor)
        {
            sampled_indices.push_back(i);
        }
    }
    pset->delete_points(sampled_indices);

    // step 1: refine planes
    std::cout << "refining planes..." << std::endl;
    const std::vector<VertexGroup::Ptr> &groups = pset->groups();
    if (groups.empty())
    {
        std::cerr << "planar segments do not exist" << std::endl;
        return EXIT_FAILURE;
    }
    HypothesisGenerator hypothesis(pset);
    hypothesis.refine_planes();

    // step 2: generate face hypothesis
    std::cout << "generating plane hypothesis..." << std::endl;
    Map *mesh = hypothesis.generate();
    if (!mesh)
    {
        std::cerr << "failed generating candidate faces. Please check if the input point cloud has good planar segments" << std::endl;
        return EXIT_FAILURE;
    }
    hypothesis.compute_confidences(mesh, false);

    // step 3: face selection
    std::cout << "optimization..." << std::endl;
    const auto &adjacency = hypothesis.extract_adjacency(mesh);
    FaceSelection selector(pset, mesh);
    std::cout << "Before optimize: " << mesh->size_of_facets() << std::endl;
    selector.optimize(adjacency, LinearProgramSolver::SCIP);
    std::cout << "After optimize: " << mesh->size_of_facets() << std::endl;
    if (mesh->size_of_facets() == 0)
    {
        std::cerr << "optimization failed: model has on faces" << std::endl;
        return EXIT_FAILURE;
    }
    // now we don't need the point cloud anymore, and it can be deleted
    delete pset;

    // step 4: save result to file
    if (MapIO::save(output_file, mesh))
    {
        // std::cout << "reconstructed model saved to file: " << output_file << std::endl;
    }
    else
    {
        // std::cerr << "failed saving reconstructed model to file: " << output_file << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
};
