$(document).onchange(function(){
    $("#viz").load("../templates/viz.html");
});

function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("choice-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("choice");
    for(i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function crowdMovies() {
    $.get('/make-crowd-movies', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
    window.location.reload();
}

function plotScalars() {
    $.get('/plot-scalar-summary', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
    window.location.reload();
}

function plotTransitionGraphs() {
    $.get('/plot-transition-graph', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
    window.location.reload();
}

function generateIndex() {
    $.get('/generate-viz-index', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
    window.location.reload();
}

function plotUsages() {
    $.get('/plot-usages', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
    window.location.reload();
}

