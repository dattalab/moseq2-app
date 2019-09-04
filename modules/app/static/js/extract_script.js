$(document).onchange(function() {
    $("#extract").load("../templates/extract.html");
});

var benchFiles = []
var openNavBool = false;

function getLocalDir() {
    $.get('/get-local-dir', {}, function(resp) {
        if (resp.ok) {
            // change button color to green and display extract button
            var fileList = resp.files;
            benchFiles = fileList;
            for(var i = 0; i < fileList.length; i++) {
                $('#filelist').append('<option>'+fileList[i]+'</option>');
            }
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
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });
}

function findROI() {
    $.get('/find-roi', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
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

function toggleBatchModeExtract(evt) {
    if (evt.currentTarget.className == "choice") {
        evt.currentTarget.className += " active";
        console.log(evt.currentTarget);
        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'extract-button') {
                console.log(d[i].id);
                document.getElementById(d[i].id).innerHTML = 'Batch Extract';
                document.getElementById(d[i].id).value = 'batch-extract';
                document.getElementById(d[i].id).style.borderRight = '3px solid orange';
            }
        }
    } else {
        evt.currentTarget.className = evt.currentTarget.className.replace(" active", "");

        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'extract-button') {
                console.log(d[i].id);
                document.getElementById(d[i].id).value = 'extract-button';
                document.getElementById(d[i].id).innerHTML = 'Extract Raw Data';
                document.getElementById(d[i].id).style.borderRight = '';
            }
        }
    }
}

function openNav() {
  if (!openNavBool) {
      document.getElementById("mySidebar").style.width = "300px";
      document.getElementById("extract-tab").style.marginRight = "300px";
      openNavBool = !openNavBool;
  } else {
      document.getElementById("mySidebar").style.width = "0";
      document.getElementById("extract-tab").style.marginRight= "0";
      openNavBool = !openNavBool;
  }
}

function closeNav() {
  document.getElementById("mySidebar").style.width = "0";
  document.getElementById("extract-tab").style.marginRight= "0";
  openNavBool = !openNavBool;
}

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
    if (evt.currentTarget.id == 'init-step-button') {
        getLocalDir();
    }
}
