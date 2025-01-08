$(window).bind("load", function () {
    results = null;
    currentTab = "papers";
    currentSort = "similarity"; // Add sorting state

    f = document.getElementById("query_field");
    f.style.height = "0px";
    f.style.height = f.scrollHeight + "px";

    // Making the textfield and placeholder act nice on iOS.
    $("#query_field").focus(function () {
        $("#query_field").addClass("placeholder_hidden");
    });

    $("#query_field").blur(function () {
        $("#query_field").removeClass("placeholder_hidden");
    });

    // Only autofocus the textfield if on desktop
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry/i.test(navigator.userAgent)) {
        $("#query_field").blur();
    } else {
        $("#query_field").focus();
    }

    // Expand textfield in accordance with text length
    $("#query_field").on("input", function () {
        this.style.height = 0;
        this.style.height = (this.scrollHeight) + "px";
    });

    // Handle toggling between papers and authors tabs
    $(".toggle").on("click", function () {
        if (this.dataset.tab == "papers") {
            togglePapersTab(true);
        } else {
            togglePeopleTab(true);
        }
    });

    // Toggle the right tab
    tabGetParameter = findGetParameter("tab");
    if (tabGetParameter != null) {
        if (tabGetParameter.toLowerCase() == "people") {
            currentTab = "people";
        } else {
            currentTab = "papers";
        }
    }

    // Insert query if present as GET parameter
    queryGetParameter = findGetParameter("q");
    if (queryGetParameter != null) {
        $("#query_field").val(queryGetParameter);
        $("#query_field").trigger("input"); // trigger resize
        performSearch();
    }

    // Listen for when user hits return
    $("#query_field").keypress(function (e) {
        if (e.which == 13) {
            performSearch();
            return false;
        }
    });

    // Initialize theme
    initializeTheme();
    
    // Check system preference changes
    if (window.matchMedia) {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        mediaQuery.addListener((e) => {
            if (!localStorage.getItem('theme')) {
                const theme = e.matches ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', theme);
                updateThemeToggleButton(theme);
            }
        });
    }
});


// Add sort toggle UI to the container div
function addSortToggle() {
    const sortToggleHtml = `
    <div id="sort_toggle_container" class="appear" style="margin-top: 10px;">
        <div class="toggle_flex">
            <div class="toggle toggle_enabled" data-sort="similarity">
                <p>Sort by Similarity</p>
            </div>
            <div class="toggle" data-sort="year">
                <p>Sort by Year</p>
            </div>
        </div>
    </div>`;
    
    $("#toggle_container").after(sortToggleHtml);
    
    // Add click handlers for sort toggle
    $("[data-sort]").on("click", function() {
        const sortType = $(this).data("sort");
        $("[data-sort]").removeClass("toggle_enabled");
        $(this).addClass("toggle_enabled");
        currentSort = sortType;
        if (results) {
            if (currentTab === "papers") {
                const sortedPapers = sortPapers(results.papers, sortType);
                addPapers(sortedPapers);
            }
        }
    });
}

function togglePapersTab(animated) {
    $('[data-tab="papers"]').first().addClass("toggle_enabled");
    $('[data-tab="people"]').first().removeClass("toggle_enabled");
    
    // Show warning if needed when switching tabs
    $("#warning_container").toggle(checkLowScores(results.papers));
    
    if (animated) {
        if ($("#results").hasClass("move_up")) {
            $("#results").removeClass("move_up");
            $("#results").on("transitionend", function () {
                const sortedPapers = sortPapers(results.papers, currentSort);
                addPapers(sortedPapers);
                $("#results").addClass("move_up");
            });
        } else {
            const sortedPapers = sortPapers(results.papers, currentSort);
            addPapers(sortedPapers);
            $("#results").addClass("move_up");
        }
    } else {
        const sortedPapers = sortPapers(results.papers, currentSort);
        addPapers(sortedPapers);
    }
    queryVal = findGetParameter("q");
    currentTab = "papers";
    updateGetParameter(queryVal, currentTab);
}

function togglePeopleTab(animated) {
    $('[data-tab="people"]').first().addClass("toggle_enabled");
    $('[data-tab="papers"]').first().removeClass("toggle_enabled");
    
    // Hide warning when switching to people tab
    $("#warning_container").hide();
    
    if (animated) {
        if ($("#results").hasClass("move_up")) {
            $("#results").removeClass("move_up");
            $("#results").on("transitionend", function () {
                addAuthors(results["authors"]);
                $("#results").addClass("move_up");
            });
        } else {
            addAuthors(results["authors"]);
            $("#results").addClass("move_up");
        }
    } else {
        addAuthors(results["authors"]);
    }
    queryVal = findGetParameter("q");
    currentTab = "people";
    updateGetParameter(queryVal, currentTab);
}

function performSearch() {
    const field = document.getElementById("query_field");
    field.style.animationName = "change_color";
    field.readOnly = true;
    $(field).blur();

    // Show loading animation
    $("#loading_container").show();
    $("#results").hide();

    let queryVal = $('textarea[name="query"]').val();

    $.getJSON("/search", { query: queryVal }, function (data) {
        console.log(data);
        field.style.animationName = "";
        field.readOnly = false;

        // Hide loading animation
        $("#loading_container").hide();

        if (data["error"] == null) {
            results = data;
            updateGetParameter(queryVal, currentTab);
            $("#error_container").hide();
            $("#warning_container").hide();
            $("#tip").hide();

            if (checkLowScores(data.papers)) {
                $("#warning_container").show();
            }
            if (!$("#sort_toggle_container").length) {
                addSortToggle();
            }
            if (currentTab == "people") {
                togglePeopleTab(true);
            } else {
                togglePapersTab(true);
            }
            $("#toggle_container").addClass("appear");
            $("#results").show();
        } else {
            $("#error_text").text(data["error"]);
            $("#error_container").show();
            $("#warning_container").hide();
        }
    });
}


// Function to sort papers
function sortPapers(papers, sortType) {
    return [...papers].sort((a, b) => {
        if (sortType === "year") {
            // First compare years
            const yearDiff = parseInt(b.year) - parseInt(a.year);
            if (yearDiff !== 0) return yearDiff;
            // If years are same, compare months
            const months = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            };
            return months[b.month] - months[a.month];
        } else {
            // Sort by similarity score (default)
            return b.score - a.score;
        }
    });
}

function findGetParameter(parameterName) {
    var result = null;
    var tmp = [];
    location.search.substr(1).split("&").forEach(function (item) {
        tmp = item.split("=");
        if (tmp[0] === parameterName) {
            result = decodeURIComponent(tmp[1]);
        }
    });
    return result;
}

function updateGetParameter(query, tab) {
    protocol = window.location.protocol + "//";
    host = window.location.host;
    pathname = window.location.pathname;
    queryParam = `?q=${encodeURIComponent(query)}`;
    tabParam = `&tab=${encodeURIComponent(tab)}`;
    newUrl = protocol + host + pathname + queryParam + tabParam;
    window.history.pushState({ path: newUrl }, '', newUrl);
}

function renderMath() {
    let config = [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
    ];
    renderMathInElement(document.body, { delimiters: config });
}

function resultClicked(e) {
    $(e).removeClass("result_clickable");
    $(e).find(".result_abstract").removeClass("truncated_text");
    $(e).find(".result_button_container").show();
}

function addPapers(data) {
    $("#results").empty();
    var html = "";
    data.forEach(e => {
        html += addPaper(e);
    });
    $("#results").append(html);
    $("#results").addClass("move_up");
    renderMath();
}

function addPaper(result) {
    let dotClass = result.score >= 0.80 ? "dot_green" : "dot_orange";
    
    // Format the authors string using our new function
    const formattedAuthors = formatAuthors(result.authors);
    
    return `<div class="search_result result_clickable" onclick="resultClicked(this)">
        <div class="result_top">
            <div class="result_year black"><p>${result.month} ${result.year}</p></div>
            <div class="result_score black" title="Cosine similarity">
                <p>${result.score}</p>
                <div class="result_dot ${dotClass}"></div>
            </div>
        </div>
        <p class="result_title black">
            ${result.title}
        </p>
        <p class="result_authors">${formattedAuthors}</p>
        <p class="result_abstract truncated_text black">${result.abstract}</p>
        <div class="result_button_container">
            <div class="result_button_flex">
                <a href="https://arxiv.org/abs/${result.id}" target="_blank">
                    <div class="result_button">
                        <div class="go_to_symbol"></div>
                        <p>Go to Paper</p>
                    </div>
                </a>
                <a href="/?q=${encodeURIComponent("https://arxiv.org/abs/" + result.id)}" target="_blank">
                    <div class="result_button">
                        <div class="similarity_symbol"></div>
                        <p>Find Similar</p>
                    </div>
                </a>
            </div>
        </div>
    </div>`;
}

function addAuthors(authors) {
    $("#results").empty();
    var html = '<div id="authors_flex">';
    authors.forEach(author => {
        // Get unique papers by ID
        const uniquePapers = Array.from(
            new Map(author.papers.map(paper => [paper.id, paper])).values()
        );
        const authorWithUniquePapers = {...author, papers: uniquePapers};
        html += addAuthor(authorWithUniquePapers);
    });
    html += '</div>';
    $("#results").append(html);
    $("#results").addClass("move_up");
    renderMath();
}

function addAuthor(author) {
    let dotClass = author.avg_score >= 0.80 ? "dot_green" : "dot_orange";
    html = `<div class="author_container">
        <div class="author_top_row">
            <p class="author_name black">${author.author}</p>
            <div class="result_score black" title="Average cosine similarity">
                <p>${author.avg_score}</p>
                <div class="result_dot ${dotClass}"></div>
            </div>
        </div>
        <div class="num_papers_container">
            <div class="author_num_papers_info_symbol" data-author="${author.author}" onmouseover="infoHover(this)" onmouseout="infoLeave()"></div>
            <p class="author_num_papers black">${author.papers.length} matching papers</p>
        </div>
        <div class="info_container" data-author="${author.author}">
            <p class="black">Based on your query, we retrieved 100 papers. Of those, <b>${author.author}</b> was (co-) author on <b>${author.papers.length}</b>.</p>
        </div>`;
    author.papers.forEach(paper => {
        // Format authors for each paper in the author view
        const formattedAuthors = formatAuthors(paper.authors);
        html += `
        <div class="author_paper_container">
            <a href="https://arxiv.org/abs/${paper.id}" class="author_paper" target="_blank">${paper.title}</a>
            <p class="author_paper_authors">${formattedAuthors}</p>
        </div>`;
    });
    html += "</div>";
    return html;
}

function infoHover(elem) {
    pos = $(elem).position();
    width = $(elem).width();
    height = $(elem).height();
    offsetLeft = $(elem).offset().left;
    infoAuthor = elem.dataset.author;
    $(".info_container").each(function(i, obj) {
        containerAuthor = obj.dataset.author;
        if (infoAuthor != containerAuthor) return;
        centerX = pos.left + width / 2;
        bottomY = pos.top + height;
        containerWidth = $(obj).width();
        left = centerX-containerWidth/2;
        minLeftMargin = 10;
        if (left < minLeftMargin) {
            diff = minLeftMargin - left;
            $(obj).css({top: bottomY+10+"px", left: left+diff/2+"px", display: "block"});
        } else {
            $(obj).css({top: bottomY+10+"px", left: left+"px", display: "block"});
        }
    });
}

function infoLeave() {
    $(".info_container").each(function(i, obj) {
        $(obj).hide();
    });
}

// Add this new function to format authors
function formatAuthors(authorString) {
    // Split authors by comma and clean up whitespace
    const authors = authorString.split(',').map(author => author.trim());
    
    // Check for collaboration
    const collaborationIndex = authors.findIndex(author => 
        author.toLowerCase().includes('collaboration'));
    
    if (collaborationIndex !== -1) {
        // Return just the collaboration name
        return authors[collaborationIndex];
    }
    
    // If no collaboration found, handle based on author count
    if (authors.length <= 5) {
        // For 1-5 authors, show all
        return authors.join(', ');
    } else {
        // For more than 5 authors, show first author + et al.
        return `${authors[0]} et al.`;
    }
}


// Theme handling
function initializeTheme() {
    // Check if user has a saved preference
    let savedTheme = localStorage.getItem('theme');
    // If no saved preference, check system preference
    if (!savedTheme) {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        savedTheme = prefersDark ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggleButton(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggleButton(newTheme);
}

function updateThemeToggleButton(theme) {
    const button = document.querySelector('.theme-toggle');
    if (theme === 'dark') {
        button.textContent = '☀️';
    } else {
        button.textContent = '🌙';
    }
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', initializeTheme);


function checkLowScores(papers) {
    // Check if any paper has a score higher than 0.2
    return !papers.some(paper => paper.score > 0.2);
}

