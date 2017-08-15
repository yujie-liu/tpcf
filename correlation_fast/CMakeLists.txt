set(correlation_fastSRC correlation_fast.cc ConfigParser.cc)
set(correlation_fastHEADER hist_fast.h ConfigParser.h)
add_executable(correlation_fast ${correlation_fastSRC} ${correlation_fastHEADER})
target_link_libraries(correlation_fast ${ROOT_LIBRARIES})

set(step2SRC step2.cc ConfigParser.cc)
set(step2HEADER hist_fast.h ConfigParser.h)
add_executable(correlation_fast_step2 ${step2SRC} ${step2HEADER})
target_link_libraries(correlation_fast_step2 ${ROOT_LIBRARIES})

set(tab_integralSRC tab_integral.cc ConfigParser.cc)
set(tab_integralHEADER hist_fast.h ConfigParser.h)
add_executable(tab_integral ${tab_integralSRC} ${tab_integralHEADER})
target_link_libraries(tab_integral ${ROOT_LIBRARIES})

install(TARGETS correlation_fast correlation_fast_step2 tab_integral DESTINATION bin)
