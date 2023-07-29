import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def set_color_alpha(rgb_val, alpha=1):
    return f'rgba({rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]}, {alpha})'


def mexican_hat(t, sigma=1.0):
    """
    Generate a Mexican hat (Ricker wavelet) function.

    Parameters:
    t : array
        Input time array.
    sigma : float, optional
        Standard deviation of the Mexican hat function. Default is 1.0.

    Returns:
    y : array
        Mexican hat function evaluated at each point in t.
    """
    t = np.asarray(t)
    y = (1 - (t / sigma) ** 2) * np.exp(-(t ** 2) / (2 * sigma ** 2))
    return y


def generate_time_series(mean, std_dev, num_samples):
    return np.random.normal(mean, std_dev, num_samples)


def generate_network(input_size, kernel_size, stride, layer_count, padding_style='valid'):
    network = []
    paddings = []

    for _ in range(layer_count):
        if padding_style == 'same':
            pad = (kernel_size - 1)
            pad_left = pad // 2
            pad_right = pad - pad_left
            padding = [True] * pad_left + [False] * input_size + [True] * pad_right
            paddings.append(padding)
            network.append(list(range(input_size + pad)))
            input_size = math.ceil(input_size / stride)
        else:  # padding_style == 'valid'
            padding = [False] * input_size
            paddings.append(padding)
            network.append(list(range(input_size)))
            input_size = math.ceil((input_size - kernel_size + 1) / stride)
    return network, paddings


def plot_time_series(fig, time_series):
    fig.add_trace(go.Scatter(x=list(range(len(time_series))), y=time_series, mode='lines+markers',
                             line=dict(color=BASE_TREND_COLOR_DIM),
                             marker=dict(size=5, color=BASE_TREND_COLOR_DIM), showlegend=False))


def generate_neuron_traces(network, distance, paddings=None):
    neuron_traces = []
    for i, layer in enumerate(network):
        offset = (len(network[0]) - len(layer)) / 2 * distance
        layer_trace = []
        padding = paddings[i]
        for j in range(len(layer)):

            if padding[j]:
                color = 'rgba(128, 128, 128, 0.2)'
            else:
                color = BASE_NODE_COLOR_DIM

            neuron = go.Scatter(x=[j * distance + offset], y=[-i], mode='markers',
                                marker=dict(size=10, color=color), showlegend=False)
            layer_trace.append(neuron)
        neuron_traces.append(layer_trace)
    return neuron_traces


def generate_receptive_field(network, stride, kernel_size, neuron_traces, fig, start_layer,
                             neuron_index, paddings, padding_style, distance):
    receptive_field = {i: [] for i in range(len(network))}
    receptive_field[start_layer] = [neuron_index]

    for i in reversed(range(start_layer)):
        for neuron in receptive_field[i + 1]:
            if padding_style == 'same':
                x = -((kernel_size - 1) // 2) * (stride - 1)
                start_neuron = max(0, (stride * neuron - (kernel_size - 1) // 2) + x)
                end_neuron = min((stride * neuron + (kernel_size - 1) // 2) + x, len(network[i]))
            else:  # padding_style == 'valid'
                start_neuron = stride * neuron
                end_neuron = min(start_neuron + kernel_size - 1, len(network[i]))
            #                 print(f'(W - F + 2P) / S + 1 = {(len(network[i]) - kernel_size + 0) / stride + 1}')

            # Ensure that the neuron index is within the bounds of the padding list
            if neuron < len(paddings[i + 1]) and not paddings[i + 1][neuron]:
                for j in range(start_neuron, end_neuron + 1):
                    offset = (len(network[0]) - len(network[i])) / 2 * distance
                    offset_next = (len(network[0]) - len(network[i + 1])) / 2 * distance

                    color = 'rgba(128, 128, 128, 0.6)' if j < len(paddings[i]) and paddings[i][
                        j] else BASE_NODE_COLOR_STICK
                    fig.add_trace(go.Scatter(x=[j * distance + offset,
                                                neuron * distance + offset_next],
                                             y=[-i, -(i + 1)], mode='lines',
                                             line=dict(color=color, width=0.5), showlegend=False))

                    # Do not add padding to the receptive field
                    if j < len(paddings[i]) and not paddings[i][j] and j not in receptive_field[i]:
                        receptive_field[i].append(j)

    return receptive_field


def update_neuron_traces_color(neuron_traces, receptive_field, paddings):
    for i, layer in enumerate(neuron_traces):
        for j, neuron_trace in enumerate(layer):
            if paddings[i][j]:
                neuron_trace['marker']['color'] = 'rgba(128, 128, 128, 0.6)'  # Grey for padding
            if j in receptive_field[i]:
                if paddings[i][j]:
                    neuron_trace['marker']['color'] = 'rgba(128, 128, 128, 0.6)'  # Grey for padding
                else:
                    neuron_trace['marker']['color'] = BASE_NODE_COLOR_M  # Blue for non-padding


def plot_receptive_field(fig, receptive_field, paddings, time_series):
    ts_indices = receptive_field[0]
    ts_indices = sorted(ts_indices)
    padding_left = paddings[0].count(True) // 2
    ts_indices = [idx - padding_left for idx in ts_indices]
    fig.add_trace(go.Scatter(x=np.arange(len(time_series))[ts_indices],
                             y=time_series[ts_indices],
                             mode='lines+markers',
                             line=dict(color=BASE_TREND_COLOR_M),
                             marker=dict(size=5, color=BASE_TREND_COLOR_M),
                             showlegend=False))


def plot_layer_labels(fig, network, distance, receptive_fract):
    text = f"Trend Covered: <span style = 'color:{BASE_TREND_COLOR_M};' ><b> {round(receptive_fract * 100, 1)} %<b></span>"
    offset = 0
    fig.add_annotation(
        x=offset - 3,  # Adjust this value to move the text horizontally
        y=+2,
        text=text,
        showarrow=False,
        font=dict(size=15,
                  color=BASE_TEXT_COLOR_G
                  )
    )

    for i, layer in enumerate(network):
        text = f"Layer {i}<br>Output" if i != 0 else "Pre-Conv.<br>Time-Series"
        offset = (len(network[0]) - len(layer)) / 2 * distance
        fig.add_annotation(
            x=offset - 3,  # Adjust this value to move the text horizontally
            y=-i,
            text=text,
            showarrow=False,
            font=dict(size=12, color=BASE_TEXT_COLOR_G)
        )
    return fig


def create_plot(time_series, stride, kernel_size, padding_style,
                layer_count, num_samples, start_layer, neuron_index):
    # """
    # It's also worth mentioning that the output size of a convolutional layer can be
    # calculated by the formula (W - F + 2P) / S + 1 where W is the input size, F is
    # the kernel size, P is the padding size, and S is the stride. In the 'valid' padding
    # scheme, P is 0 because we only consider positions where the kernel and the input
    # fully overlap. So the formula simplifies
    # """
    # (W - F + 2P) / S + 1

    fig = go.Figure()
    network, paddings = generate_network(input_size=len(time_series), kernel_size=kernel_size,
                                         stride=stride, layer_count=layer_count + 1,
                                         padding_style=padding_style)

    distance = len(time_series) / (len(network[0]) - 1)
    plot_time_series(fig, time_series)
    neuron_traces = generate_neuron_traces(network, distance, paddings)
    # Inside the main code
    receptive_field = generate_receptive_field(network, stride=stride,
                                               kernel_size=kernel_size,
                                               neuron_traces=neuron_traces,
                                               fig=fig, start_layer=start_layer,
                                               neuron_index=neuron_index, paddings=paddings,
                                               padding_style=padding_style, distance=distance)

    update_neuron_traces_color(neuron_traces, receptive_field, paddings)

    for layer in neuron_traces:
        for neuron_trace in layer:
            fig.add_trace(neuron_trace)

    plot_receptive_field(fig, receptive_field, paddings, time_series)

    receptive_fract = len(receptive_field[0]) / len(time_series)
    fig = plot_layer_labels(fig, network, distance, receptive_fract)

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),
        plot_bgcolor='white',
        autosize=False,
        width=1000,

        margin=go.layout.Margin(
            l=20,  # left margin
            r=20,  # right margin
            b=0,  # bottom margin
            t=2,  # top margin
            pad=5  # padding
        ),
    )
    return fig


def generate_code(layer_count, kernel_size, stride, padding_style):
    layer_text = ''
    for i in range(layer_count):
        adder = ', input_shape=(None, 1)' if i == 0 else ''
        ender = '' if layer_count == (i + 1) else '\n    '
        layer_text += f"model.add(tf.keras.layers.Conv1D(filters=1, kernel_size={kernel_size}, strides={stride}, padding='{padding_style}'{adder})){ender}"
    return layer_text


BASE_NODE_COLOR = [17, 150, 97]
BASE_TREND_COLOR = [255, 75, 75]
BASE_TEXT_COLOR_G = [54, 54, 54]

BASE_TEXT_COLOR_G = set_color_alpha(BASE_TEXT_COLOR_G, alpha=1)
BASE_NODE_COLOR_M = set_color_alpha(BASE_NODE_COLOR, alpha=1)
BASE_NODE_COLOR_STICK = set_color_alpha(BASE_NODE_COLOR, alpha=0.6)
BASE_NODE_COLOR_DIM = set_color_alpha(BASE_NODE_COLOR, alpha=0.2)

BASE_TREND_COLOR_M = set_color_alpha(BASE_TREND_COLOR, alpha=1)
BASE_TREND_COLOR_DIM = set_color_alpha(BASE_TREND_COLOR, alpha=0.2)

TITLE = 'Receptive Field Generator'
IM_CONSTANTS = {'LOGO': "https://i.ibb.co/fn617h9/Capture-removebg-preview-1.png"}

CODE_TEXT = """
import tensorflow as tf

def build_model():
    model = tf.keras.models.Sequential()
    # Add the Conv1D layers
    {}
    return model

model = build_model()
"""

HTML_X = """
<style>
.reportview-container .main .block-container {
    padding-top: -200px;
}
</style>
"""

HTML_Y = """
<style>
       .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 15rem;
            padding-right: 15rem;
        }
</style>
"""


def init_time_series(mean=1, std=0.02, wavelet_width=5, samples=27):
    t_wavelet = np.linspace(-wavelet_width, wavelet_width, 2 * wavelet_width + 1)
    wavelet = mexican_hat(t_wavelet, sigma=wavelet_width / 2)  # Generate the time series
    time_series = generate_time_series(mean=mean, std_dev=std, num_samples=int(samples))
    # Select a random starting index for the wavelet
    start_idx = np.random.randint(0, len(time_series) - len(wavelet) + 1)
    # Add the wavelet to the time series at the selected location
    time_series[start_idx:start_idx + len(wavelet)] += wavelet
    return time_series


def main():
    # format page
    st.set_page_config(TITLE, page_icon=IM_CONSTANTS['LOGO'], layout='wide')
    st.markdown(HTML_Y, unsafe_allow_html=True)
    st.title(TITLE)
    default_samples = 27
    if 'num_samples' not in st.session_state:
        st.session_state['time_series'] = init_time_series(samples=default_samples)
    # setting up the header with its own row
    _, r1_col1, r1_col2, r1_col3, _ = st.columns([1, 4, 1, 6, 1])
    with r1_col1:
        st.markdown(f"<p style='font-size: 27px;'><i>How Parameters Effect NN Propagation</i></p>",
                    unsafe_allow_html=True)
    with r1_col3:
        st.write('')
    # main information line: includes map location
    r2_col0, r2_col1, r2_col2, r2_col3, _ = st.columns([1, 3, 1, 10, 1])
    with r2_col1:
        if 'num_samples' not in st.session_state:
            st.session_state['num_samples'] = default_samples
        # Add sliders for kernel_size and stride
        num_samples = st.slider('Select Time-Series Length', min_value=10, max_value=150, value=int(default_samples))
        if num_samples != int(st.session_state['num_samples']):
            # Generate the time series
            st.session_state['time_series'] = init_time_series(samples=num_samples)
            st.session_state['num_samples'] = num_samples

        kernel_size = st.slider('Select kernel size', min_value=1, max_value=10, value=5)
        stride = st.slider('Select stride', min_value=1, max_value=10, value=2)
        layer_count = st.slider('Select layer count', min_value=1, max_value=5, value=2)
        # Add radio buttons for padding_style
        padding_style = st.radio('Select padding style', ['valid', 'same'])
        start_layer = st.slider('Select start layer', min_value=1, max_value=layer_count, value=2)
        network, paddings = generate_network(input_size=len(st.session_state['time_series']), kernel_size=kernel_size,
                                             stride=stride, layer_count=layer_count + 1,
                                             padding_style=padding_style)

        p_num_delta = np.invert(np.array(paddings[start_layer] * 1)).sum()
        p_num_min = (len(paddings[start_layer]) - p_num_delta) // 2
        print([p_num_min, p_num_delta])
        neuron_index = st.slider('Select neuron index', min_value=int(p_num_min),  max_value=int(p_num_min + p_num_delta - 1), value=3)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
    # white space
    with r2_col2:
        st.write("")
    # plot container
    with r2_col3:
        layer_text = generate_code(layer_count, kernel_size, stride, padding_style)
        code = CODE_TEXT.format(layer_text)
        st.code(code, language='python')

        if st.session_state['time_series'] is not None:
            fig = create_plot(st.session_state['time_series'], stride, kernel_size, padding_style,
                              layer_count, num_samples, start_layer, neuron_index)
            st.plotly_chart(fig, use_container_width=True)
        empty_col, empty_col, img_col = st.columns([6, 1, 2])  # Adjust these numbers as needed
        st.markdown(HTML_X, unsafe_allow_html=True)
        img_col.image(IM_CONSTANTS['LOGO'], width=110)


if __name__ == "__main__":
    main()
