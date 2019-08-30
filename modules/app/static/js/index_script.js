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