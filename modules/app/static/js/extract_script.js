$(document).onchange(function() {
    $("#extract").load("../templates/extract.html");
});

var datapath = '/';

function checkLocalDir() {
    $.get('/check-local-dir', {}, function(resp) {
        if (resp.ok) {
            // change button color to green and display extract button
            datapath = resp.message;
            alert(resp.message);
            document.getElementById('dir-check').checked = true;
            document.getElementById('dir-check2').checked = true;
        } else {
            alert(resp.message);
        }
    });
}

function genLocalConfig() {
    $.get('/generate-config', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            document.getElementById('gen-exists').checked = true;
            document.getElementById('gen-exists2').checked = true;
        } else {
            alert(resp.message);
        }
    });
}

function extractRaw() {
    $.get('/extract-raw', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}

function findROI() {
    $.get('/find-roi', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}

function showButtons() {
    var flipFiles = document.getElementsByClassName('flip-file');
    for(var i = 0; i < flipFiles.length; i++) {
        flipFiles[i].hidden = false;
    }
}

function downloadFlip(event) {
    var clickedId = event.target.id;
    $.get('/download-flip-file', {"flip-id": clickedId}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}


/*

*/



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
