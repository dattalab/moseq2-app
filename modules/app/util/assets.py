from flask_assets import Bundle, Environment
from .. import app

bundles = {

    'index_js': Bundle(
        'js/index_script.js',
        output='gen/index_script.js',
        filters='jsmin'),

    'index_css': Bundle(
        'lib/reset.css',
        'css/index_style.css',
        output='gen/index_style.css',
        filters='cssmin'),

    'pca_js': Bundle(
        'js/pca_script.js',
        output='gen/pca_script.js',
        filters='jsmin'),

    'pca_css': Bundle(
        'lib/reset.css',
        'css/pca_style.css',
        output='gen/pca_style.css',
        filters='cssmin'),

    'model_js': Bundle(
        'js/model_script.js',
        output='gen/model_script.js',
        filters='jsmin'),

    'model_css': Bundle(
        'lib/reset.css',
        'css/model_style.css',
        output='gen/model_style.css',
        filters='cssmin'),

    'viz_js': Bundle(
        'js/viz_script.js',
        output='gen/viz_script.js',
        filters='jsmin'),

    'viz_css': Bundle(
        'lib/reset.css',
        'css/viz_style.css',
        output='gen/viz_style.css',
        filters='cssmin')
}

assets = Environment(app)

assets.register(bundles)