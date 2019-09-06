$(document).onload(function(){
    $("#pca").load("../templates/pca.html");
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

function loadPCA() {
    alert('clicked');
}

function trainPCA() {
    $.get('/train-pca', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });

}

function clipScores() {
    $.get('/clip-pca-scores', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });
}

function applyPCA() {
    $.get('/apply-pca', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });
}

function computeCP() {
    $.get('/compute-changepoints', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });
}