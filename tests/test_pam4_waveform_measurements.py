def test_oma():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("tests/SSPRQ_sequence.csv")
    ssprq = np.array(df.iloc[:,0])
    ssprq = np.tile(ssprq, 5)  # Make 10x longer, for better metric averaging

    np.random.seed(1123125)

    datarate_gbps = 112
    samples_per_symbol = 32
    symbol_time = 1 / (datarate_gbps * 1e9)
    dt_sec = symbol_time / samples_per_symbol

    # Initialize eye
    eye = IdealEye(
        datarate_gbps=datarate_gbps,
        dt_sec=dt_sec,
        format="PAM4",
        sampling_offset_mode="adaptive"
    )

    # Make a sample waveform
    oma = 1.0723
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=0,  # this is not used, if bit_pattern argument is provided
        amplitude=oma,
        std=0.2,
        dt_sec=dt_sec,
        bw_3db_Hz=0.58*datarate_gbps*1e9,  # 0.58
        npoles=4,
        format="PAM4",
        bit_pattern=(ssprq / 3) * oma,
    )
    eye.add_data(wvf, "mV")

    m = eye.get_measurements()
    print(f"TDECQ = {m['tdecq_outer']} dB")
    print(f"SER = {m['vertical_ser']}")
    print(f"rise time (10-90): {m['rise_time_10-90_4140']} ps")
    print(f"fall time (90-10): {m['fall_time_90-10_4140']} ps")

    # eye.plot(show=True, pattern=[3]*3)
    # eye.plot(show=True, pattern=[2]*3)
    # eye.plot(show=True, pattern=[1]*3)
    # eye.plot(show=True, pattern=[0]*3)
    # eye.plot(show=True, pattern=[0,1,0])
    # eye.plot(show=True, pattern=[3,0,3])
    # eye.plot(show=True, pattern=[3,2,0])

    eye.plot(show=False)
    plt.close()

    # eye.plot_bathtub(show=True, bathtub_index=0)
    # eye.plot_bathtub(show=True, bathtub_index=1)
    # eye.plot_bathtub(show=True, bathtub_index=2)

    # Make sure plots can be generated
    eye.plot_bathtub_cross_section(show=False, direction="vertical", y_axis="ber")
    plt.close()
    eye.plot_bathtub_cross_section(show=False, direction="vertical", y_axis="q-scale")
    plt.close()
    eye.plot_bathtub_cross_section(show=False, direction="horizontal", y_axis="ber")
    plt.close()
    eye.plot_bathtub_cross_section(show=False, direction="horizontal", y_axis="q-scale")
    plt.close()


# def test_ser_correlation():
#     from stateye import IdealEye
#     from sample_signals.generate_signals import generate_data_with_filtered_noise
#     import numpy as np
#     import math
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm
#     import pandas as pd

#     np.random.seed(1123125)

#     df = pd.read_csv("tests/SSPRQ_sequence.csv")
#     ssprq = np.array(df.iloc[:,0])
#     ssprq = np.tile(ssprq, 5)  # Make 10x longer, for better metric averaging

#     datarate_gbps = 112
#     samples_per_symbol = 32
#     symbol_time = 1 / (datarate_gbps * 1e9)
#     dt_sec = symbol_time / samples_per_symbol

#     oma_values = np.linspace(0.5, 1.0, 10)
#     bw_fraction = np.linspace(0.5, 0.7, 6)

#     ser = []
#     oma_outer = []
#     tdecq = []

#     for bwf in tqdm(bw_fraction):
#         ser += [[]]
#         oma_outer += [[]]
#         tdecq += [[]]
#         for oma in tqdm(oma_values):
#             # Initialize eye
#             eye = IdealEye(
#                 datarate_gbps=datarate_gbps,
#                 dt_sec=dt_sec,
#                 format="PAM4",
#                 # num_bits_to_filter_on_before=0,
#                 # num_bits_to_filter_on_after=0,
#                 sampling_offset_mode="adaptive"
#             )

#             # Make a sample waveform
#             wvf = generate_data_with_filtered_noise(
#                 samples_per_symbol=samples_per_symbol,
#                 nbits=0,
#                 amplitude=oma,
#                 std=0.1,
#                 dt_sec=dt_sec,
#                 bw_3db_Hz=bwf*datarate_gbps*1e9,  # 0.58
#                 npoles=4,
#                 format="PAM4",
#                 bit_pattern=(ssprq / 3) * oma,
#             )
#             eye.add_data(wvf, "mW")

#             m = eye.get_measurements()
#             print(f"OMA_outer = {m['oma_outer']} mV")
#             print(f"TDECQ = {m['tdecq_outer']} dB")
#             print(f"SER = {m['vertical_ser']}")
#             print(f"rise time (10-90): {m['rise_time_10-90_4140']} ps")
#             print(f"fall time (90-10): {m['fall_time_90-10_4140']} ps")

#             oma_outer[-1] += [10*np.log10(m['oma_outer'])]
#             ser[-1] += [m['vertical_ser']]
#             tdecq[-1] += [m['tdecq_outer']]

#     # oma_tdecq = np.array(oma_outer) - np.array(tdecq)
#     # ser = np.array(ser)
#     # plt.plot(oma_tdecq, np.log10(ser), 'o')
#     # plt.xlabel("OMA - TDECQ [dBm]")
#     # plt.ylabel("log10(SER)")
#     # plt.grid(True)
#     # plt.show()

#     import json
#     with open("tests/tdecq_test.json", "w") as f:
#         json.dump(
#             {
#                 "tdecq": tdecq,
#                 "oma_outer": oma_outer,
#                 "ser": ser,
#                 "bw_fraction": bw_fraction.tolist()
#             }, f, indent=4
#         )