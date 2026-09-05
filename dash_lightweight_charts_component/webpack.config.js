const path = require('path');

module.exports = {
    entry: path.resolve(__dirname, 'src/lib/index.js'),
    output: {
        path: path.resolve(__dirname, "dash_lightweight_charts"),
        filename: 'dash_lightweight_charts.min.js',
        library: 'dash_lightweight_charts',
        libraryTarget: 'window',
    },
    // Dash's renderer already loads React/ReactDOM/PropTypes on the page and
    // exposes them as globals precisely so third-party components don't ship
    // (and clash with) their own copies. lightweight-charts is NOT externalized:
    // it's bundled in directly, so the component works without any extra
    // runtime script tags or CDN dependency.
    externals: {
        react: 'React',
        'react-dom': 'ReactDOM',
        'prop-types': 'PropTypes',
    },
    module: {
        rules: [
            {
                test: /\.jsx?$/,
                exclude: /node_modules/,
                use: 'babel-loader',
            },
        ],
    },
    resolve: {
        extensions: ['.js', '.jsx'],
    },
    mode: 'production',
    devtool: false,
};
