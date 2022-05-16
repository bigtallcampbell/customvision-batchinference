# customvision-batchinference
Shows how to use Azure Custom Vision's Python SDK to run Object Detection on a series of images.

# Prerequisites
1. From your project, go to Settings and get the Project ID (will be a GUID), Training Key (Guid without dashes), and Endpoint (URL).
    ![Project ID](/images/project_id.png 'Project ID Locations')
1. Go to Performance and get the **published** name of the iteration you'd like to use.  
    ![Iteration Name](/images/iteration_name.png 'Iteration Name')
1. Plug those vales into the params.json file

# Running the Sample
Open with VSCode, build the container, double check the params.json file, and F5/run the project.  Results will be in the `output` folder


