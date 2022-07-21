#include <iostream>
#include <vector>

#include "H5Cpp.h"


#define DLLEXPORT extern "C" __declspec(dllexport)


DLLEXPORT double convert_digital_to_analog(int value, int max_analog_value, int min_analog_value, int max_digital_value, int min_digital_value) {
    double digital_value = min_analog_value + value * (max_analog_value - min_analog_value) / (max_digital_value - min_digital_value);
    // clean saturated values
    if ((digital_value > 4095) || (digital_value < -4095)) {
        digital_value = 0;
    }
    return digital_value;
}


DLLEXPORT void process_raw_data_chunk(int* data_chunk, int recording_length, int* ch_to_extract, double** rec, int max_analog_value, int min_analog_value, int max_digital_value, int min_digital_value) {
  int nb_channel = sizeof(ch_to_extract) / sizeof(ch_to_extract[0]);

  for(int frame_nb=0; frame_nb<recording_length; frame_nb++) {
    int frame_start_id = frame_nb * nb_channel;

    for(int ch_id=0; ch_id<nb_channel; ch_id++) {
      int ch = ch_to_extract[ch_id];
      double digital_value = convert_digital_to_analog(data_chunk[frame_start_id + ch], max_analog_value, min_analog_value, max_digital_value, min_digital_value);
      rec[ch_id][frame_nb] = digital_value;
    } // end for ch_id
  } // end for frame_nb
} // end process_raw_data_chunk


DLLEXPORT void read_raw_data(char* path, double t_start, double t_end, int* ch_to_extract, int frame_chunk, double* recording) {

  H5::H5File file(path, H5::H5F_ACC_RDWR);
  DataSet data = file.openDataSet("Well_A1");

}
