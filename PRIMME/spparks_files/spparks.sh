#!/bin/bash
#replacement_text = [num_processor, in_file_name]
fileBase="spparks"  #Sets base name for log, dump, cluster, and jpeg files
mpirun -np ##1## ../../SPPARKS/src/spk_lambda -var fileBase $fileBase < ##2##.in
