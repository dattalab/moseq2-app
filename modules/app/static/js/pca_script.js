$(document).onload(function(){
    $("#pca").load("../templates/pca.html");
});

function openTabPCA(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("choice-contentP");
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

    var url = '/train-pca';
    var formData = new FormData();

    $('#slice-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('depth-file', filename);
    });

    $('#params input, #params select').each(
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
            alert(response.files.split(' || '));
        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
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