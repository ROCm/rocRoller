#pragma once

#include <map>
#include <string>
#include <vector>

class Arg
{
public:
    Arg() = delete;
    Arg(std::vector<std::string> flags, std::string usage);

    ~Arg() = default;

    std::vector<std::string> flags();
    std::string              usage();

private:
    std::vector<std::string> m_flags;
    std::string              m_usage;
};

class ParseOptions
{
public:
    ParseOptions();
    explicit ParseOptions(std::string helpMessage);

    ~ParseOptions() = default;

    void validateArgs();
    void addArg(std::string name, Arg const& arg);
    void parse_args(int argc, const char* argv[]);
    void print_help();

    template <typename T>
    T get(std::string name, T const& defaultVal);

private:
    std::map<std::string, std::string> m_parsed_args;
    std::map<std::string, Arg>         m_valid_args;
    std::string                        m_helpMessage;
};
