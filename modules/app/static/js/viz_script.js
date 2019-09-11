$(document).onchange(function(){
    $("#viz").load("../templates/viz.html");
});

$(document).onload = function() {
    document.getElementById("defaultOpen").click();
};

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

function openGraph(evt, graphName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("viz-tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("viz-tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(graphName).style.display = "block";
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
    var url = '/generate-viz-index';
    var formData = new FormData();

    $('#index-params input, #index-params select').each(
        function(index){
            var input = $(this);
            if (formData.get(input.attr('name'))) {
                var prev = formData.get(input.attr('name'));
                let newVal = [prev, input.val()];
                formData.set(input.attr('name'), newVal);
            } else{
                formData.append(input.attr('name'), input.val());
            }
        }
    );

    fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(response => {

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
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

