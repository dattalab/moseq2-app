$(document).ready(function () {
    $(document).on("scroll", onScroll);
    $("#extract").load("../templates/extract.html");
    $("#pca").load("../templates/pca.html");
    $("#model").load("../templates/model.html");
    $("#viz").load("../templates/viz.html");

    //smoothscroll
    $('a[href^="#"]').on('click', function (e) {
        e.preventDefault();
        $(document).off("scroll");

        $('a').each(function () {
            $(this).removeClass('active');
        });

        $(this).addClass('active');
        var target = this.hash,
            menu = target;
        $target = $(target);
        $('html, body').stop().animate({
            'scrollTop': $target.offset().top-55
        }, 100, 'swing', function () {
            window.location.hash = target;
            $(document).on("scroll", onScroll);
        });
    });
});

function displayParamModal(ev) {
    // Get the modal
    var modal = document.getElementById("paramModal");

    // Get the button that opens the modal
    var btn = document.getElementById("param-op");

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[1];

    // When the user clicks the button, open the modal
    modal.style.display = "block";

    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
      modal.style.display = "none";
    }

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
      }
    }
}

function displayHelpModal(ev) {
    // Get the modal
    var modal = document.getElementById("helperModal");

    // Get the button that opens the modal
    var btn = document.getElementById("help-op");

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];

    // When the user clicks the button, open the modal
    modal.style.display = "block";

    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
      modal.style.display = "none";
    }

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
      }
    }
}

function onScroll(event){
    var scrollPos = $(document).scrollTop();
    $('#horizontal-list a').each(function () {
        var currLink = $(this);
        var refElement = $(currLink.attr("href"));
        if (refElement.position().top-70 <= scrollPos && refElement.position().top-70 + refElement.height() > scrollPos) {
            $('#horizontal-list ul li a').removeClass("active");
            currLink.addClass("active");
        }
        else{
            currLink.removeClass("active");
        }
    });
}