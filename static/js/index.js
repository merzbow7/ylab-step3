const select = document.querySelector('.form-select')
const form = document.querySelector('.geo-form')

function eventHandler(event) {
    form.submit()
    // console.log(event.target.value)
    // console.log(event.target)
    // if (event.target.value === '') {
    //     form.submit()
    // }

}

select.addEventListener('change', eventHandler)