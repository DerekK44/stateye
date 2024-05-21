def test_oma():
    from stateye import IdealEye
    from sample_signals.generate_signals import generate_data_with_filtered_noise
    import numpy as np
    import math
    import matplotlib.pyplot as plt

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
        # num_bits_to_filter_on_before=0,
        # num_bits_to_filter_on_after=0,
        sampling_offset_mode="adaptive"
    )

    # Make a sample waveform
    oma = 1.0723
    wvf = generate_data_with_filtered_noise(
        samples_per_symbol=samples_per_symbol,
        nbits=100000,
        amplitude=oma,
        std=0.2,
        dt_sec=dt_sec,
        bw_3db_Hz=0.55*datarate_gbps*1e9,  # 0.58
        npoles=4,
        format="PAM4",
    )
    eye.add_data(wvf, "mV")

    m = eye.get_measurements()
    print(f"TDECQ = {m['tdecq_outer']} dB")
    print(f"SER = {m['vertical_ser']}")

    # eye.plot(show=True, pattern=[3]*3)
    # eye.plot(show=True, pattern=[2]*3)
    # eye.plot(show=True, pattern=[1]*3)
    # eye.plot(show=True, pattern=[0]*3)
    # eye.plot(show=True, pattern=[0,1,0])
    # eye.plot(show=True, pattern=[3,0,3])
    # eye.plot(show=True, pattern=[3,2,0])
    eye.plot(show=True)

    # eye.plot_bathtub(show=True, bathtub_index=0)
    # eye.plot_bathtub(show=True, bathtub_index=1)
    # eye.plot_bathtub(show=True, bathtub_index=2)

    eye.plot_bathtub_cross_section(show=True, direction="vertical", y_axis="ber")
    eye.plot_bathtub_cross_section(show=True, direction="vertical", y_axis="q-scale")
    eye.plot_bathtub_cross_section(show=True, direction="horizontal", y_axis="ber")
    eye.plot_bathtub_cross_section(show=True, direction="horizontal", y_axis="q-scale")

    # for oma_type in ["xp", "4140", "8180"]:
    #     measured_oma = eye.get_measurements()[f"oma_{oma_type}"]
    #     assert math.isclose(oma, measured_oma, rel_tol=1e-2)