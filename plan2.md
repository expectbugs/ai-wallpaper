IMPLEMENT SD3.5 LARGEST BEST MODEL

research good LoRAs for SD3.5 specifically, for as many categories as possible

download them into the organized directories, make a sd3.5 specific path

**DIFFERENT PIPELINE**

Have SD 3.5 generate 1024x1024 tiles in a cohesive and as big-picture-aware manner as possible as though the image was generated in one fell swoop, to get as close to the target resolution as possible without going over in either direction.

Fill the difference between actual and target resolution with Outpainting.

Run a Refinement pass to fix any artifacts or coherance differences or outpainting seams.

Save image.

Make sure there is support for saving images at each stage with the --save-stages option.

We're going to see if this results in richer images then the SDXL way of doing it.