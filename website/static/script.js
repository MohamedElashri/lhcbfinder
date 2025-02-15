// Global variables
let results = null;
let currentSort = "similarity";
let currentPage = 1;
let totalPages = 1;
let resultsPerPage = 10;
let sortedPapers = []; // Array to store the sorted papers

$(window).bind("load", function () {
    // Initialize theme first before any other operations
    initializeTheme();

    const f = document.getElementById("query_field");
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

    // Insert query if present as GET parameter
    const queryGetParameter = findGetParameter("q");
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

    // Add viewport height fix
    setViewportHeight();
    window.addEventListener('resize', setViewportHeight);
    
    // Better textarea handling for mobile
    const queryField = document.getElementById("query_field");
    queryField.addEventListener('input', () => adjustTextareaHeight(queryField));
    
    // Disable animations on mobile if reduced motion is preferred
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        document.documentElement.classList.add('reduce-motion');
    }
    
    // Handle mobile keyboard appearance
    if (isMobileDevice()) {
        window.addEventListener('resize', function() {
            if (document.activeElement.tagName === 'TEXTAREA') {
                window.scrollTo(0, 0);
            }
        });
    }
});

function addPageNumber(container, pageNum) {
    const pageBtn = $("<div>")
        .addClass("page-number")
        .text(pageNum)
        .on("click", () => {
            // Update currentPage AFTER creating the button but BEFORE addPapers is called
            currentPage = pageNum;

            // Add 'active' class to the clicked page button
            $(".page-number").removeClass("active"); // Remove from all
            $(pageBtn).addClass("active"); // Add to the clicked button

            addPapers(sortedPapers);
        });

    // Check if this button is for the current page and add 'active' class if so
    if (pageNum === currentPage) {
        pageBtn.addClass("active");
    }

    container.append(pageBtn);
}

function addEllipsis(container) {
    container.append($("<span>").addClass("pagination-ellipsis").text("..."));
}

function addPaginationControls() {
    const paginationHtml = `
        <div id="results_controls">
            <div id="results_per_page_flex">
                <label for="results_per_page" class="results-label">Results per page:</label>
                <select id="results_per_page" class="form-select">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="15">15</option>
                    <option value="25">25</option>
                    <option value="50">50</option>
                </select>
            </div>
            <div id="pagination_flex">
                <button id="prev_page" class="page-btn">Previous</button>
                <div id="page_numbers"></div>
                <button id="next_page" class="page-btn">Next</button>
            </div>
        </div>`;
    $("#sort_toggle_container").after(paginationHtml);

    // Add event listeners
    $("#results_per_page").on("change", function() {
        resultsPerPage = parseInt($(this).val());
        currentPage = 1;  // Reset to first page when changing results per page
        
        // Recalculate totalPages based on the new resultsPerPage
        totalPages = Math.ceil(sortedPapers.length / resultsPerPage);

        updatePagination();
        addPapers(sortedPapers);
    });

    $("#prev_page").on("click", () => {
        if (currentPage > 1) {
            currentPage--;
            updatePagination();
            addPapers(sortedPapers);
        }
    });

    $("#next_page").on("click", () => {
        if (currentPage < totalPages) {
            currentPage++;
            updatePagination();
            addPapers(sortedPapers);
        }
    });
}

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
        currentPage = 1;
        sortPapers(currentSort);

        // Recalculate totalPages after sorting
        totalPages = Math.ceil(sortedPapers.length / resultsPerPage);

        updatePagination();
        addPapers(sortedPapers);
    });
}

function performSearch() {
    const field = document.getElementById("query_field");
    field.style.animationName = "change_color";
    field.readOnly = true;
    $(field).blur();

    $("#loading_container").show();
    $("#results").hide();

    let queryVal = $('textarea[name="query"]').val();

    $.getJSON("/search", {
        query: queryVal
    }, function(data) {
        field.style.animationName = "";
        field.readOnly = false;
        $("#loading_container").hide();

        if (data["error"] == null) {
            results = data;
            // Initialize sortedPapers with the received papers, unsorted
            sortedPapers = [...data.papers];
            // Sort the papers based on the current sort setting
            sortPapers(currentSort);
            updateGetParameter(queryVal); // Remove tab parameter
            $("#error_container").hide();
            $("#warning_container").hide();
            $("#tip").hide();

            if (checkLowScores(data.papers)) {
                $("#warning_container").show();
            }

            if (!$("#sort_toggle_container").length) {
                addSortToggle();
            }

            if (!$("#results_controls").length) {
                addPaginationControls();
            }

            // Update pagination based on the total results
            totalPages = Math.ceil(data.total_results / resultsPerPage);
            updatePagination();

            // Display the results based on the current tab
            addPapers(sortedPapers);

            $("#results").show();
        } else {
            $("#error_text").text(data["error"]);
            $("#error_container").show();
            $("#warning_container").hide();
        }
    });

    if (isMobileDevice()) {
        document.activeElement.blur(); // Hide mobile keyboard
        window.scrollTo(0, 0); // Scroll to top for results
    }
}

function updatePagination() {
    const prevButton = $("#prev_page");
    const nextButton = $("#next_page");
    const pageNumbers = $("#page_numbers");

    // Correctly update button states based on currentPage and totalPages
    prevButton.prop("disabled", currentPage <= 1); // Disable if on the first page
    nextButton.prop("disabled", currentPage >= totalPages); // Disable if on the last page

    // Clear existing page numbers
    pageNumbers.empty();

    if (totalPages <= 7) {
        // Show all pages if total is 7 or less
        for (let i = 1; i <= totalPages; i++) {
            addPageNumber(pageNumbers, i);
        }
    } else {
        // Logic for paginating when totalPages > 7
        addPageNumber(pageNumbers, 1); // Always show the first page

        if (currentPage > 4) {
            addEllipsis(pageNumbers);
        }

        const startPage = Math.max(2, currentPage - 1);
        const endPage = Math.min(totalPages - 1, currentPage + 1);

        for (let i = startPage; i <= endPage; i++) {
            addPageNumber(pageNumbers, i);
        }

        if (currentPage < totalPages - 3) {
            addEllipsis(pageNumbers);
        }

        addPageNumber(pageNumbers, totalPages); // Always show the last page
    }
}

function sortPapers(sortType) {
    if (!results || !results.papers) return;

    if (sortType === "year") {
        sortedPapers = [...results.papers].sort((a, b) => {
            const yearDiff = parseInt(b.year) - parseInt(a.year);
            if (yearDiff !== 0) return yearDiff;
            const months = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            };
            return months[b.month] - months[a.month];
        });
    } else {
        sortedPapers = [...results.papers].sort((a, b) => b.score - a.score);
    }
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

function updateGetParameter(query) {
    const protocol = window.location.protocol + "//";
    const host = window.location.host;
    const pathname = window.location.pathname;
    const queryParam = `?q=${encodeURIComponent(query)}`;
    const newUrl = protocol + host + pathname + queryParam;
    window.history.pushState({ path: newUrl }, '', newUrl);
}

function renderMath() {
    const config = [
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

function addPapers(papers) {
    $("#results").empty();
    const start = (currentPage - 1) * resultsPerPage;
    const end = start + resultsPerPage;
    const papersToDisplay = papers.slice(start, end);
    let html = "";
    papersToDisplay.forEach(paper => {
        html += addPaper(paper);
    });
    $("#results").append(html);
    $("#results").addClass("move_up");
    renderMath();
}

function addPaper(result) {
    let dotClass = result.score >= 0.80 ? "dot_green" : "dot_orange";
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

function formatAuthors(authorString) {
    const authors = authorString.split(',').map(author => author.trim());
    
    const collaborationIndex = authors.findIndex(author => 
        author.toLowerCase().includes('collaboration'));
    
    if (collaborationIndex !== -1) {
        return authors[collaborationIndex];
    }
    
    if (authors.length <= 5) {
        return authors.join(', ');
    } else {
        return `${authors[0]} et al.`;
    }
}

// Theme handling
function initializeTheme() {
    let savedTheme = localStorage.getItem('theme');
    if (!savedTheme) {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        savedTheme = prefersDark ? 'dark' : 'light';
        localStorage.setItem('theme', savedTheme); // Save the initial theme
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
    button.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function checkLowScores(papers) {
    return !papers.some(paper => paper.score > 0.2);
}

// Add mobile detection and handling
function isMobileDevice() {
    return (
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
        (navigator.maxTouchPoints && navigator.maxTouchPoints > 2)
    );
}

// Update textarea height calculation
function adjustTextareaHeight(textarea) {
    textarea.style.height = '0';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// Add viewport height fix for mobile browsers
function setViewportHeight() {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
}